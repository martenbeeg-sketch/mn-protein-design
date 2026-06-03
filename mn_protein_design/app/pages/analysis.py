from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mn_protein_design.app.components.molstar_viewer import (
    ChainVisualization,
    StructureVisualization,
    molstar_custom_component,
)
from mn_protein_design.app.pages.common import result_link
from mn_protein_design.core.candidates import STAGE_COMPLEX_REFOLDING
from mn_protein_design.core.jobs import collect_jobs, read_json
from mn_protein_design.workflows.analysis import DEFAULT_THRESHOLDS, run_analysis_contract
from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates


st.title("Analysis")
st.caption("Ranks and filters complex-refolded candidates using normalized AF2/Boltz2 metrics.")

sources = [
    source
    for source in candidate_sources()
    if STAGE_COMPLEX_REFOLDING in source["stage_counts"]
]
if not sources:
    st.info("No complex-refolded candidate sets are available yet.")
    st.stop()

labels = [
    f"{source['task_group']} / {source['job_code']} | {source['tool']} | {source['candidate_count']} candidates"
    for source in sources
]
selected = st.selectbox("Candidate set", range(len(sources)), format_func=lambda index: labels[index])
source = sources[selected]
candidates = [
    candidate
    for candidate in load_source_candidates(source)
    if candidate.get("stage") == STAGE_COMPLEX_REFOLDING
]


def _candidate_result_kind(candidate: dict) -> str:
    metrics = candidate.get("metrics") or {}
    raw_metadata = candidate.get("raw_metadata") or {}
    explicit = metrics.get("result_kind") or raw_metadata.get("result_kind")
    if explicit:
        return str(explicit)
    tool = str(candidate.get("source_tool") or source.get("tool") or "").lower()
    if tool in {
        "bindcraft",
        "boltzgen",
        "pxdesign",
        "proteina_complexa",
        "genie3",
        "protpardelle_1c",
        "rfdiffusion3_foundry",
        "rfdiffusion_classic",
    }:
        return "native_pipeline"
    if metrics.get("analysis_backend") or candidate.get("stage") == "analysis":
        return "app_reanalysis"
    return "normalized"


def _native_table_specs(source_run_dir: Path, source_tool: str) -> list[tuple[str, Path]]:
    raw_dir = source_run_dir / "artifacts" / "raw"
    tool = source_tool.lower()
    specs: list[tuple[str, Path]] = []
    if tool == "bindcraft":
        output_dir = raw_dir / "bindcraft" / "output"
        for name, label in [
            ("final_design_stats.csv", "BindCraft final design stats"),
            ("trajectory_stats.csv", "BindCraft trajectory stats"),
            ("mpnn_design_stats.csv", "BindCraft MPNN design stats"),
        ]:
            path = output_dir / name
            if path.exists():
                specs.append((label, path))
    elif tool == "boltzgen":
        ranked_dir = raw_dir / "boltzgen" / "run-vanilla" / "final_ranked_designs"
        all_metrics = ranked_dir / "all_designs_metrics.csv"
        if all_metrics.exists():
            specs.append(("BoltzGen all designs metrics", all_metrics))
        for path in sorted(ranked_dir.glob("final_designs_metrics_*.csv")):
            specs.append((f"BoltzGen {path.name}", path))
    elif tool == "pxdesign":
        px_dir = raw_dir / "pxdesign"
        for path in sorted([*px_dir.glob("output/design_outputs/*/summary.csv"), *px_dir.glob("design_outputs/*/summary.csv")]):
            specs.append((f"PXDesign summary: {path.parent.name}", path))
    elif tool == "proteina_complexa":
        eval_dir = raw_dir / "proteina_complexa" / "evaluation_results"
        for path in sorted(eval_dir.glob("*/RAW_protein_binder_results_search_binder_local_pipeline_combined.csv")):
            specs.append(("Proteina-Complexa combined binder results", path))
        for path in sorted(eval_dir.glob("*/binder_results_search_binder_local_pipeline_*.csv")):
            specs.append((f"Proteina-Complexa binder results: {path.stem}", path))
        for path in sorted(eval_dir.glob("*/overall_binder_performance_search_binder_local_pipeline_aggregated.csv")):
            specs.append(("Proteina-Complexa aggregated performance", path))
    elif tool == "genie3":
        output_dir = raw_dir / "genie3" / "output"
        for path in sorted(output_dir.glob("*/results/info.csv")):
            specs.append((f"Genie3 info: {path.parents[1].name}", path))
        for path in sorted(output_dir.glob("*/results/v0_success/success_info.csv")):
            specs.append((f"Genie3 v0 successes: {path.parents[2].name}", path))
        for path in sorted(output_dir.glob("*/results/v0_success/successful_incomplex_binders_cluster.csv")):
            specs.append((f"Genie3 successful binder clusters: {path.parents[2].name}", path))
    elif tool == "protpardelle_1c":
        output_dir = raw_dir / "protpardelle_1c" / "output"
        for path in sorted(output_dir.glob("**/esm_metrics.csv")):
            specs.append((f"Protpardelle-1c ESMFold metrics: {path.parent.name}", path))
        for path in sorted(output_dir.glob("**/scaffold_info.csv")):
            specs.append((f"Protpardelle-1c scaffold info: {path.parent.name}", path))
        for path in sorted(output_dir.glob("**/design_input.csv")):
            specs.append((f"Protpardelle-1c design input: {path.parent.name}", path))
    elif tool == "rfdiffusion3_foundry":
        output_dir = raw_dir / "rfdiffusion3_foundry"
        mapping = output_dir / "foundry_native_mapping.tsv"
        if mapping.exists():
            specs.append(("Foundry native design mapping", mapping))
        for path in sorted(output_dir.glob("rf3/**/*_ranking_scores.csv")):
            specs.append((f"RF3 ranking scores: {path.parent.name}", path))
        for path in sorted(output_dir.glob("rf3/**/*_confidences.csv")):
            specs.append((f"RF3 confidences: {path.parent.name}", path))
    return [(label, path) for label, path in specs if path.is_file()]


def _read_native_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def _proteina_complexa_native_pass_df(df: pd.DataFrame) -> pd.Series | None:
    required = {"self_complex_i_pAE", "self_complex_pLDDT"}
    rmsd_column = "self_binder_scRMSD_ca" if "self_binder_scRMSD_ca" in df.columns else "self_binder_scRMSD"
    if not required.issubset(df.columns) or rmsd_column not in df.columns:
        return None
    ipae = pd.to_numeric(df["self_complex_i_pAE"], errors="coerce")
    plddt = pd.to_numeric(df["self_complex_pLDDT"], errors="coerce")
    rmsd = pd.to_numeric(df[rmsd_column], errors="coerce")
    return (ipae * 31.0 <= 7.0) & (plddt >= 0.9) & (rmsd < 1.5)


def _augment_proteina_complexa_native_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "self_complex_i_pAE" in df.columns:
        ipae = pd.to_numeric(df["self_complex_i_pAE"], errors="coerce")
        df["native_ipae_scaled"] = ipae * 31.0
        df = df.assign(_native_rank_value=ipae, _native_rank_missing=ipae.isna())
        df = df.sort_values(["_native_rank_missing", "_native_rank_value"], ascending=[True, True]).drop(
            columns=["_native_rank_value", "_native_rank_missing"]
        )
        df.insert(0, "native_rank", range(1, len(df) + 1))
        df["native_rank_metric"] = "self_complex_i_pAE"
    elif "self_complex_i_pTM" in df.columns:
        iptm = pd.to_numeric(df["self_complex_i_pTM"], errors="coerce")
        df = df.assign(_native_rank_value=iptm, _native_rank_missing=iptm.isna())
        df = df.sort_values(["_native_rank_missing", "_native_rank_value"], ascending=[True, False]).drop(
            columns=["_native_rank_value", "_native_rank_missing"]
        )
        df.insert(0, "native_rank", range(1, len(df) + 1))
        df["native_rank_metric"] = "self_complex_i_pTM"
    native_pass = _proteina_complexa_native_pass_df(df)
    if native_pass is not None:
        df["native_pass_filters"] = native_pass
    return df.reset_index(drop=True)


def _native_structure_text(path: Path) -> str:
    return gzip.open(path, "rt", errors="ignore").read() if path.name.endswith(".gz") else path.read_text(errors="ignore")


def _native_candidate_structure_path(source_run_dir: Path, candidate: dict) -> Path | None:
    structure_text = str(candidate.get("complex_pdb") or candidate.get("binder_pdb") or "").strip()
    if not structure_text:
        return None
    structure_path = Path(structure_text)
    if structure_path.is_absolute():
        return structure_path if structure_path.exists() else None
    resolved = source_run_dir / structure_path
    return resolved if resolved.exists() else None


def _truthy_native_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "passed", "accepted"}:
        return True
    if text in {"false", "0", "no", "failed", "rejected"}:
        return False
    return None


def _native_candidate_label(index: int, candidate: dict) -> str:
    metrics = candidate.get("metrics") or {}
    label_parts = [str(index), str(candidate.get("candidate_id") or "candidate")]
    final_rank = metrics.get("final_rank") or metrics.get("native_final_rank")
    if final_rank not in {None, ""}:
        label_parts.append(f"rank {final_rank}")
    pass_filters = _truthy_native_value(metrics.get("native_pass_filters"))
    if pass_filters is None:
        pass_filters = _truthy_native_value(metrics.get("pass_filters"))
    if pass_filters is not None:
        label_parts.append("passing" if pass_filters else "not passing")
    return " | ".join(label_parts)


def _native_candidate_sort_key(candidate: dict) -> tuple:
    metrics = candidate.get("metrics") or {}
    source_tool = str(candidate.get("source_tool") or "").lower()
    if source_tool == "protpardelle_1c":
        def number(key: str, default: float) -> float:
            try:
                return float(metrics.get(key))
            except (TypeError, ValueError):
                return default

        return (
            0 if _truthy_native_value(metrics.get("native_pass_filters")) is True else 1,
            number("ca_scaffold_scrmsd", float("inf")),
            number("pae", float("inf")),
            -number("binder_plddt", float("-inf")),
            number("allatom_motif_pred_rmsd", float("inf")),
        )
    final_rank = metrics.get("final_rank") or metrics.get("native_final_rank")
    try:
        return (0, float(final_rank))
    except (TypeError, ValueError):
        return (1, str(candidate.get("candidate_id") or ""))


def _show_native_design_viewer(source_run_dir: Path, source_candidates: list[dict]) -> None:
    structure_options = [
        (index, candidate, structure_path)
        for index, candidate in enumerate(sorted(source_candidates, key=_native_candidate_sort_key), start=1)
        for structure_path in [_native_candidate_structure_path(source_run_dir, candidate)]
        if structure_path is not None
    ]
    if not structure_options:
        st.info("No native structure file could be resolved for these candidates.")
        return

    st.subheader("Native Design Viewer")
    selected_index = st.selectbox(
        "Native design",
        range(len(structure_options)),
        format_func=lambda option_index: _native_candidate_label(
            structure_options[option_index][0],
            structure_options[option_index][1],
        ),
    )
    _display_index, candidate, structure_path = structure_options[selected_index]
    target_chains = [str(chain) for chain in candidate.get("target_chains") or [] if str(chain)]
    binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
    chain_visualizations = [
        *[
            ChainVisualization(
                chain_id=chain,
                color="uniform",
                color_params={"value": "0xb8bec9"},
                representation_type="cartoon",
                label=f"Target {chain}",
            )
            for chain in target_chains
        ],
        *[
            ChainVisualization(
                chain_id=chain,
                color="uniform",
                color_params={"value": "0x1f6feb"},
                representation_type="cartoon+ball-and-stick",
                label=f"Binder {chain}",
            )
            for chain in binder_chains
        ],
    ]
    molstar_custom_component(
        structures=[
            StructureVisualization(
                pdb=_native_structure_text(structure_path),
                color="chain-id",
                representation_type="cartoon",
                chains=chain_visualizations,
            )
        ],
        key=f"native_pipeline_viewer_{source_run_dir.name}_{selected_index}_{structure_path.name}",
        height=620,
        show_controls=True,
    )


def _show_native_pipeline_results(source_run_dir: Path, source_tool: str, source_candidates: list[dict]) -> None:
    st.subheader("Native Pipeline Results")
    specs = _native_table_specs(source_run_dir, source_tool)
    if not specs:
        st.caption("No native tool result table was found for this candidate set. Use App Re-analysis below for normalized ranking.")
        _show_native_design_viewer(source_run_dir, source_candidates)
        return
    if len(specs) == 1:
        selected_label, selected_path = specs[0]
        st.caption(selected_label)
    else:
        selected_index = st.selectbox(
            "Native result table",
            range(len(specs)),
            format_func=lambda index: specs[index][0],
        )
        selected_label, selected_path = specs[selected_index]
    df = _read_native_csv(selected_path)
    if df.empty:
        st.info("The selected native result table is empty.")
        return
    if source_tool.lower() == "proteina_complexa":
        df = _augment_proteina_complexa_native_df(df)
    summary_cols = st.columns(4)
    summary_cols[0].metric("Rows", len(df))
    pass_columns = [
        col
        for col in df.columns
        if col.lower() in {"native_pass_filters", "pass_filters", "passes_filters", "accepted"} or col.lower().endswith("-success")
    ]
    if pass_columns:
        pass_col = pass_columns[0]
        pass_values = df[pass_col].astype(str).str.lower().isin({"true", "1", "yes", "passed", "accepted"})
        summary_cols[1].metric(f"Passing by {pass_col}", int(pass_values.sum()))
    if source_tool.lower() == "proteina_complexa" and "self_complex_i_pAE" in df.columns:
        values = pd.to_numeric(df["self_complex_i_pAE"], errors="coerce")
        if values.notna().any():
            summary_cols[2].metric("Best self_complex_i_pAE", f"{values.min():.3g}")
    elif source_tool.lower() == "protpardelle_1c" and "plddt" in df.columns:
        values = pd.to_numeric(df["plddt"], errors="coerce")
        if values.notna().any():
            display_value = values.max() * 100.0 if values.max() <= 1.0 else values.max()
            summary_cols[2].metric("Best pLDDT", f"{display_value:.3g}")
    elif source_tool.lower() == "protpardelle_1c" and "pae" in df.columns:
        values = pd.to_numeric(df["pae"], errors="coerce")
        if values.notna().any():
            summary_cols[2].metric("Best PAE", f"{values.min():.3g}")
    else:
        score_columns = [col for col in df.columns if col.lower() in {"quality_score", "analysis_score", "score", "af2_iptm", "iptm"}]
        if score_columns:
            values = pd.to_numeric(df[score_columns[0]], errors="coerce")
            if values.notna().any():
                summary_cols[2].metric(f"Best {score_columns[0]}", f"{values.max():.3g}")
    if source_tool.lower() == "proteina_complexa" and "native_ipae_scaled" in df.columns:
        values = pd.to_numeric(df["native_ipae_scaled"], errors="coerce")
        if values.notna().any():
            summary_cols[3].metric("Best native iPAE*31", f"{values.min():.3g}")
    elif "score_columns" in locals() and score_columns:
        values = pd.to_numeric(df[score_columns[0]], errors="coerce")
        if values.notna().any():
            summary_cols[3].metric("Mean score", f"{values.mean():.3g}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Native file: {selected_path.relative_to(source_run_dir)}")
    _show_native_design_viewer(source_run_dir, source_candidates)


rows = []
for candidate in candidates:
    metrics = candidate.get("metrics") or {}
    row = {
        "candidate_id": candidate.get("candidate_id"),
        "stage": candidate.get("stage"),
        "source_tool": candidate.get("source_tool"),
        "result_kind": _candidate_result_kind(candidate),
        "native_pass_filters": metrics.get("native_pass_filters", metrics.get("pass_filters")),
        "native_final_rank": metrics.get("final_rank") or metrics.get("native_final_rank"),
        "complex_pdb": candidate.get("complex_pdb"),
        "binder_sequence": candidate.get("binder_sequence"),
        "metric_count": len(metrics),
    }
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row.setdefault(key, value)
    rows.append(row)

st.subheader("Normalized Candidate Table")
normalized_df = pd.DataFrame(rows)
summary_columns = [
    "candidate_id",
    "source_tool",
    "result_kind",
    "native_pass_filters",
    "native_final_rank",
    "binder_plddt",
    "confidence",
    "iptm",
    "ipae",
    "binder_rmsd",
    "complex_pdb",
    "binder_sequence",
    "metric_count",
]
st.dataframe(normalized_df[[col for col in summary_columns if col in normalized_df.columns]], use_container_width=True, hide_index=True)
with st.expander("All normalized candidate fields"):
    st.dataframe(normalized_df, use_container_width=True, hide_index=True)

native_end_to_end_tools = {
    "bindcraft",
    "boltzgen",
    "pxdesign",
    "proteina_complexa",
    "genie3",
    "protpardelle_1c",
    "rfdiffusion3_foundry",
    "rfdiffusion_classic",
}
source_tool = str(source.get("tool") or "").lower()
if source_tool in native_end_to_end_tools:
    _show_native_pipeline_results(Path(str(source["run_dir"])), source_tool, candidates)
    st.stop()


def _analysis_run_dir_from_csv(path: Path) -> Path:
    return path.parents[2]


def _resolve_analysis_structure(path: Path, row: pd.Series) -> Path | None:
    structure_text = str(row.get("complex_pdb") or "").strip()
    if not structure_text or structure_text.lower() == "nan":
        return None
    structure_path = Path(structure_text)
    if structure_path.is_absolute() and structure_path.exists():
        return structure_path
    analysis_run_dir = _analysis_run_dir_from_csv(path)
    input_payload = read_json(analysis_run_dir / "input.json")
    source_run_dir = Path(str((input_payload.get("inputs") or {}).get("source_run_dir") or ""))
    candidates = [
        source_run_dir / structure_path,
        analysis_run_dir / structure_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _analysis_candidates_by_id(path: Path) -> dict[str, dict]:
    result = read_json(_analysis_run_dir_from_csv(path) / "result.json")
    return {str(candidate.get("candidate_id")): candidate for candidate in (result.get("outputs") or {}).get("candidates", [])}


def _analysis_candidate_for_row(path: Path, row: pd.Series) -> dict | None:
    row_id = str(row.get("candidate_id") or "")
    candidates = list(_analysis_candidates_by_id(path).values())
    by_id = {str(candidate.get("candidate_id") or ""): candidate for candidate in candidates}
    if row_id in by_id:
        return by_id[row_id]
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id.startswith(f"{row_id}_ranking_"):
            return candidate
        source_candidate = (candidate.get("raw_metadata") or {}).get("source_candidate")
        if isinstance(source_candidate, dict) and str(source_candidate.get("candidate_id") or "") == row_id:
            return candidate
    return None


def _path_from(base: Path, text: object) -> Path | None:
    if not text:
        return None
    path = Path(str(text))
    if path.is_absolute() and str(path).startswith("/work/"):
        return base / path.relative_to("/work")
    return path if path.is_absolute() else base / path


def _first_existing_path(base: Path, *values: object) -> Path | None:
    for value in values:
        path = _path_from(base, value)
        if path and path.exists():
            return path
    return None


def _structure_text(path: Path) -> str:
    return gzip.open(path, "rt", errors="ignore").read() if path.name.endswith(".gz") else path.read_text(errors="ignore")


def _monomer_overlay_paths(path: Path, row: pd.Series) -> tuple[Path | None, Path | None]:
    candidate = _analysis_candidate_for_row(path, row)
    if not candidate:
        return None, None
    analysis_run_dir = _analysis_run_dir_from_csv(path)
    analysis_input = read_json(analysis_run_dir / "input.json")
    complex_run_dir = Path(str((analysis_input.get("inputs") or {}).get("source_run_dir") or ""))
    complex_input = read_json(complex_run_dir / "input.json")
    monomer_run_dir = Path(str((complex_input.get("inputs") or {}).get("source_run_dir") or ""))

    complex_source = (candidate.get("raw_metadata") or {}).get("source_candidate")
    if not isinstance(complex_source, dict):
        complex_source = candidate
    native_metrics = complex_source.get("metrics") or {}
    native_reference = _first_existing_path(
        complex_run_dir,
        native_metrics.get("generation_filepath"),
        native_metrics.get("monomer_refolding_reference"),
        (complex_source.get("raw_metadata") or {}).get("input_complex"),
    )
    if native_reference:
        return native_reference, None
    monomer_source = (complex_source.get("raw_metadata") or {}).get("source_candidate")
    if not isinstance(monomer_source, dict):
        return None, None
    monomer_raw = monomer_source.get("raw_metadata") or {}
    design_run_dir = Path(str(monomer_raw.get("upstream_source_run_dir") or ""))
    reference = _path_from(design_run_dir, (monomer_source.get("metrics") or {}).get("monomer_refolding_reference"))
    if not reference or not reference.exists():
        reference = _path_from(design_run_dir, monomer_raw.get("input_complex"))
    monomer_prediction = _path_from(monomer_run_dir, monomer_source.get("complex_pdb"))
    return reference if reference and reference.exists() else None, monomer_prediction if monomer_prediction and monomer_prediction.exists() else None


def _parse_pdb_atoms(path: Path, chain_id: str | None = "A") -> list[dict]:
    atoms: list[dict] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip() or "_"
        if chain_id is not None and chain != chain_id:
            continue
        try:
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
        except ValueError:
            continue
        atoms.append(
            {
                "name": line[12:16].strip(),
                "resname": line[17:20].strip() or "UNK",
                "chain": chain,
                "resseq": line[22:26].strip(),
                "icode": line[26].strip(),
                "coord": coord,
                "element": (line[76:78].strip() if len(line) >= 78 else line[12:16].strip()[0]).upper(),
            }
        )
    return atoms


def _parse_cif_atoms(path: Path, chain_id: str | None = "A") -> list[dict]:
    atoms: list[dict] = []
    atom_headers: list[str] = []
    in_atom_loop = False
    text = gzip.open(path, "rt", errors="ignore").read() if path.name.endswith(".gz") else path.read_text(errors="ignore")
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
        if not in_atom_loop or not line.startswith(("ATOM ", "HETATM ")):
            continue
        parts = line.split()
        if len(parts) < len(atom_headers):
            continue
        row = dict(zip(atom_headers, parts))
        chain = row.get("auth_asym_id") or row.get("label_asym_id") or "_"
        if chain_id is not None and chain != chain_id:
            continue
        try:
            coord = np.array([float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])], dtype=float)
        except (KeyError, ValueError):
            continue
        atoms.append(
            {
                "name": row.get("label_atom_id") or row.get("auth_atom_id") or "X",
                "resname": row.get("auth_comp_id") or row.get("label_comp_id") or "UNK",
                "chain": chain,
                "resseq": row.get("auth_seq_id") or row.get("label_seq_id") or "1",
                "icode": "",
                "coord": coord,
                "element": row.get("type_symbol") or "X",
            }
        )
    return atoms


def _atoms_for_structure(path: Path, chain_id: str | None = "A") -> list[dict]:
    return _parse_cif_atoms(path, chain_id) if path.suffix.lower() == ".cif" or path.name.endswith(".cif.gz") else _parse_pdb_atoms(path, chain_id)


def _chain_roles_for_row(path: Path, row: pd.Series) -> tuple[str, str]:
    candidate = _analysis_candidate_for_row(path, row)
    if not candidate:
        return "A", "B"
    binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
    target_chains = [str(chain) for chain in candidate.get("target_chains") or [] if str(chain)]
    return (binder_chains[0] if binder_chains else "A", target_chains[0] if target_chains else "B")


def _first_chain(payload: dict | None, key: str, default: str) -> str:
    if not isinstance(payload, dict):
        return default
    chains = payload.get(key)
    if isinstance(chains, str) and chains:
        return chains
    if isinstance(chains, list):
        for chain in chains:
            if str(chain):
                return str(chain)
    return default


def _overlay_chain_roles(path: Path, row: pd.Series) -> tuple[str, str, str, str]:
    candidate = _analysis_candidate_for_row(path, row) or {}
    complex_source = (candidate.get("raw_metadata") or {}).get("source_candidate")
    if not isinstance(complex_source, dict):
        complex_source = candidate
    monomer_source = (complex_source.get("raw_metadata") or {}).get("source_candidate")
    reference_source = None
    if isinstance(monomer_source, dict):
        reference_source = (monomer_source.get("raw_metadata") or {}).get("source_candidate")
    if not isinstance(reference_source, dict):
        reference_source = monomer_source if isinstance(monomer_source, dict) else complex_source
    reference_binder_chain = _first_chain(reference_source, "binder_chains", _first_chain(monomer_source, "binder_chains", "A"))
    reference_target_chain = _first_chain(reference_source, "target_chains", _first_chain(monomer_source, "target_chains", "B"))
    complex_binder_chain = _first_chain(complex_source, "binder_chains", _first_chain(candidate, "binder_chains", "A"))
    complex_target_chain = _first_chain(complex_source, "target_chains", _first_chain(candidate, "target_chains", "B"))
    return reference_binder_chain, reference_target_chain, complex_binder_chain, complex_target_chain


def _kabsch_transform(reference: np.ndarray, model: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if reference.shape != model.shape or reference.shape[0] < 3:
        return None
    ref_center = reference.mean(axis=0)
    model_center = model.mean(axis=0)
    ref = reference - ref_center
    mob = model - model_center
    covariance = mob.T @ ref
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = math.copysign(1.0, np.linalg.det(u_mat @ vt_mat))
    rotation = u_mat @ correction @ vt_mat
    translation = ref_center - model_center @ rotation
    return rotation, translation


def _atoms_to_pdb(atoms: list[dict], chain_id: str, transform: tuple[np.ndarray, np.ndarray] | None = None) -> str:
    lines: list[str] = []
    rotation, translation = transform if transform is not None else (None, None)
    residue_map: dict[tuple[str, str], int] = {}
    next_residue = 1
    for atom_index, atom in enumerate(atoms, start=1):
        residue_key = (str(atom["resseq"]), str(atom.get("icode") or ""))
        if residue_key not in residue_map:
            residue_map[residue_key] = next_residue
            next_residue += 1
        coord = atom["coord"]
        if rotation is not None and translation is not None:
            coord = coord @ rotation + translation
        element = str(atom.get("element") or "X")[:2].rjust(2)
        lines.append(
            f"ATOM  {atom_index:5d} {str(atom['name'])[:4]:>4s} {str(atom['resname'])[:3]:>3s} {chain_id:1s}"
            f"{residue_map[residue_key]:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
            f"  1.00 50.00          {element}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def _atoms_to_pdb_by_chain(
    atoms: list[dict],
    chain_map: dict[str, str],
    transform: tuple[np.ndarray, np.ndarray] | None = None,
) -> str:
    lines: list[str] = []
    rotation, translation = transform if transform is not None else (None, None)
    residue_map: dict[tuple[str, str, str], int] = {}
    next_residue_by_chain: dict[str, int] = {}
    atom_index = 1
    for atom in atoms:
        source_chain = str(atom.get("chain") or "_")
        output_chain = chain_map.get(source_chain)
        if output_chain is None:
            continue
        residue_key = (output_chain, str(atom["resseq"]), str(atom.get("icode") or ""))
        if residue_key not in residue_map:
            next_residue = next_residue_by_chain.get(output_chain, 1)
            residue_map[residue_key] = next_residue
            next_residue_by_chain[output_chain] = next_residue + 1
        coord = atom["coord"]
        if rotation is not None and translation is not None:
            coord = coord @ rotation + translation
        element = str(atom.get("element") or "X")[:2].rjust(2)
        lines.append(
            f"ATOM  {atom_index:5d} {str(atom['name'])[:4]:>4s} {str(atom['resname'])[:3]:>3s} {output_chain:1s}"
            f"{residue_map[residue_key]:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
            f"  1.00 50.00          {element}"
        )
        atom_index += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def _aligned_monomer_overlay(
    reference_path: Path | None,
    monomer_path: Path | None,
    reference_binder_chain: str = "A",
    monomer_chain: str = "A",
) -> tuple[str | None, str | None]:
    if not reference_path or not monomer_path:
        return None, None
    reference_atoms = _atoms_for_structure(reference_path, reference_binder_chain)
    monomer_atoms = _atoms_for_structure(monomer_path, monomer_chain)
    if not monomer_atoms and monomer_chain != "A":
        monomer_atoms = _atoms_for_structure(monomer_path, "A")
    reference_ca = [atom for atom in reference_atoms if atom["name"] == "CA"]
    monomer_ca = [atom for atom in monomer_atoms if atom["name"] == "CA"]
    count = min(len(reference_ca), len(monomer_ca))
    if count < 3:
        return None, None
    reference = np.array([atom["coord"] for atom in reference_ca[:count]], dtype=float)
    model = np.array([atom["coord"] for atom in monomer_ca[:count]], dtype=float)
    transform = _kabsch_transform(reference, model)
    if transform is None:
        return None, None
    return _atoms_to_pdb(reference_atoms, "C"), _atoms_to_pdb(monomer_atoms, "D", transform=transform)


def _aligned_complex_overlay(
    reference_path: Path | None,
    complex_path: Path | None,
    reference_target_chain: str = "B",
    complex_binder_chain: str = "A",
    complex_target_chain: str = "B",
) -> str | None:
    if not reference_path or not complex_path:
        return None
    reference_target_atoms = _atoms_for_structure(reference_path, reference_target_chain)
    predicted_target_atoms = _atoms_for_structure(complex_path, complex_target_chain)
    reference_ca = [atom for atom in reference_target_atoms if atom["name"] == "CA"]
    predicted_ca = [atom for atom in predicted_target_atoms if atom["name"] == "CA"]
    count = min(len(reference_ca), len(predicted_ca))
    if count < 3:
        return None
    reference = np.array([atom["coord"] for atom in reference_ca[:count]], dtype=float)
    model = np.array([atom["coord"] for atom in predicted_ca[:count]], dtype=float)
    transform = _kabsch_transform(reference, model)
    if transform is None:
        return None
    predicted_atoms = _atoms_for_structure(complex_path, None)
    return _atoms_to_pdb_by_chain(predicted_atoms, {complex_binder_chain: "E", complex_target_chain: "F"}, transform=transform)


def _show_selected_design_structure(path: Path, df: pd.DataFrame, title: str) -> None:
    candidates = df[df["passes_filters"].fillna(False)] if "passes_filters" in df else df
    if candidates.empty:
        candidates = df
    structure_rows = []
    for index, row in candidates.iterrows():
        structure_path = _resolve_analysis_structure(path, row)
        if structure_path and structure_path.exists():
            structure_rows.append((index, row, structure_path))
    if not structure_rows:
        st.info("No resolved complex structure file is available for the displayed candidates.")
        return

    st.subheader("Selected Design")
    labels = [
        f"{int(row.get('analysis_rank')) if pd.notna(row.get('analysis_rank')) else index} | {row.get('candidate_id')}"
        for index, row, _structure_path in structure_rows
    ]
    selected = st.selectbox(
        "Structure",
        range(len(structure_rows)),
        format_func=lambda index: labels[index],
        key=f"{title}_structure_select",
    )
    _index, row, structure_path = structure_rows[selected]
    metric_specs = [
        (["monomer_rmsd"], "Monomer RMSD"),
        (["binder_rmsd", "complex_scrmsd"], "Binder Pose RMSD"),
        (["ipae", "source_ipae", "min_interface_pae", "avg_interface_pae"], "iPAE"),
        (["ipsae", "source_ipsae", "ipsae_min"], "ipSAE"),
    ]
    if str(row.get("source_tool") or "").lower() == "boltzgen" or "min_interaction_pae" in row:
        metric_specs = [
            (["pass_filters"], "Native pass"),
            (["final_rank"], "Native rank"),
            (["quality_score"], "Quality score"),
            (["interaction_pae"], "Interaction PAE"),
            (["min_interaction_pae"], "Min interaction PAE"),
            (["iptm"], "ipTM"),
            (["design_to_target_iptm"], "Design-target ipTM"),
            (["filter_rmsd"], "Filter RMSD"),
        ]
    metric_cols = st.columns(min(4, len(metric_specs)))
    for index, (metric_keys, label) in enumerate(metric_specs):
        col = metric_cols[index % len(metric_cols)]
        value = None
        for metric in metric_keys:
            candidate_value = row.get(metric)
            if not pd.isna(candidate_value):
                value = candidate_value
                break
        if pd.isna(value):
            metric_cols_value = "n/a"
        elif isinstance(value, bool):
            metric_cols_value = "yes" if value else "no"
        else:
            try:
                metric_cols_value = f"{float(value):.3f}"
            except (TypeError, ValueError):
                metric_cols_value = str(value)
        col.metric(label, metric_cols_value)
    toggle_cols = st.columns(2)
    with toggle_cols[0]:
        show_predicted_complex = st.checkbox(
            "Show predicted complex aligned on target",
            value=True,
            key=f"{title}_show_predicted_complex",
        )
    with toggle_cols[1]:
        show_aligned_monomer = st.checkbox(
            "Show predicted monomer aligned on backbone",
            value=True,
            key=f"{title}_show_aligned_monomer",
        )

    reference_path, monomer_path = _monomer_overlay_paths(path, row)
    if not reference_path:
        binder_chain, target_chain = _chain_roles_for_row(path, row)
        st.caption("Showing accepted complex directly; no upstream designed-backbone overlay is available for this candidate.")
        molstar_custom_component(
            structures=[
                StructureVisualization(
                    pdb=_structure_text(structure_path),
                    color="chain-id",
                    representation_type="cartoon",
                    chains=[
                        ChainVisualization(
                            chain_id=target_chain,
                            color="uniform",
                            color_params={"value": "0xb8bec9"},
                            representation_type="cartoon",
                            label="Target",
                        ),
                        ChainVisualization(
                            chain_id=binder_chain,
                            color="uniform",
                            color_params={"value": "0x1f6feb"},
                            representation_type="cartoon+ball-and-stick",
                            label="Accepted binder",
                        ),
                    ],
                )
            ],
            key=f"{title}_direct_complex_viewer_{selected}_{structure_path.name}",
            height=620,
            show_controls=True,
        )
        return
    reference_binder_chain, reference_target_chain, complex_binder_chain, complex_target_chain = _overlay_chain_roles(path, row)
    reference_text = _structure_text(reference_path)
    backbone_pdb, aligned_monomer_pdb = _aligned_monomer_overlay(
        reference_path,
        monomer_path,
        reference_binder_chain=reference_binder_chain,
        monomer_chain="A",
    )
    aligned_complex_pdb = _aligned_complex_overlay(
        reference_path,
        structure_path,
        reference_target_chain=reference_target_chain,
        complex_binder_chain=complex_binder_chain,
        complex_target_chain=complex_target_chain,
    )
    structures = []
    overlay_labels = ["starting target is gray", "designed backbone is green"]
    structures.append(
        StructureVisualization(
            pdb=reference_text,
            color="chain-id",
            representation_type="cartoon",
            chains=[
                ChainVisualization(
                    chain_id=reference_binder_chain,
                    color="uniform",
                    color_params={"value": "0x2da44e"},
                    representation_type="cartoon",
                    label="Designed backbone",
                ),
                ChainVisualization(
                    chain_id=reference_target_chain,
                    color="uniform",
                    color_params={"value": "0xb8bec9"},
                    representation_type="cartoon",
                    label="Starting target",
                ),
            ],
        )
    )
    if show_predicted_complex and aligned_complex_pdb:
        overlay_labels.extend(["predicted complex binder is blue", "predicted complex target is light gray"])
        structures.append(
            StructureVisualization(
                pdb=aligned_complex_pdb,
                color="chain-id",
                representation_type="cartoon",
                chains=[
                    ChainVisualization(
                        chain_id="E",
                        color="uniform",
                        color_params={"value": "0x1f6feb"},
                        representation_type="cartoon+ball-and-stick",
                        label="Predicted complex binder",
                    ),
                    ChainVisualization(
                        chain_id="F",
                        color="uniform",
                        color_params={"value": "0xd8dee9"},
                        representation_type="cartoon",
                        label="Predicted complex target",
                    ),
                ],
            )
        )
    if aligned_monomer_pdb and show_aligned_monomer:
        overlay_labels.append("aligned monomer refold is magenta")
        structures.append(
            StructureVisualization(
                pdb=aligned_monomer_pdb,
                color="uniform",
                color_params={"value": "0xbf3989"},
                representation_type="cartoon",
                chains=[
                    ChainVisualization(
                        chain_id="D",
                        color="uniform",
                        color_params={"value": "0xbf3989"},
                        representation_type="cartoon",
                        label="Aligned monomer refold",
                    )
                ],
            )
        )
    if show_predicted_complex and not aligned_complex_pdb:
        st.warning("Could not align the predicted complex onto the starting target for this candidate.")
    if show_aligned_monomer and not aligned_monomer_pdb and monomer_path:
        st.warning("Could not align the predicted monomer onto the designed backbone for this candidate.")
    st.caption("Overlay: " + ", ".join(overlay_labels) + ".")
    molstar_custom_component(
        structures=structures,
        key=f"{title}_structure_viewer_{selected}_{structure_path.name}",
        height=620,
        show_controls=True,
    )


def _passes_live_filters(row: pd.Series, thresholds: dict[str, float]) -> tuple[bool, str]:
    failures: list[str] = []
    checks = [
        ("binder_plddt", ">=", thresholds["min_binder_plddt"]),
        ("confidence", ">=", thresholds["min_confidence"]),
        ("iptm", ">=", thresholds["min_iptm"]),
        ("ipsae", ">=", thresholds["min_ipsae"]),
        ("ipae", "<=", thresholds["max_ipae"]),
        ("ipde", "<=", thresholds["max_ipde"]),
        ("monomer_rmsd", "<=", thresholds["max_binder_rmsd"]),
        ("binder_rmsd", "<=", thresholds["max_binder_rmsd"]),
        ("hotspot_contact_fraction", ">=", thresholds.get("min_hotspot_contact_fraction")),
        ("min_binder_to_hotspot_distance", "<=", thresholds.get("max_hotspot_distance")),
    ]
    for metric, op, threshold in checks:
        if threshold is None:
            continue
        if metric not in row or pd.isna(row.get(metric)):
            continue
        value = float(row[metric])
        if op == ">=" and value < threshold:
            failures.append(f"{metric} < {threshold:g}")
        if op == "<=" and value > threshold:
            failures.append(f"{metric} > {threshold:g}")
    return not failures, "; ".join(failures)


def _apply_live_filters(df: pd.DataFrame, thresholds: dict[str, float], keep_top_n: int) -> pd.DataFrame:
    filtered = df.copy()
    if filtered.empty:
        return filtered
    outcomes = filtered.apply(lambda row: _passes_live_filters(row, thresholds), axis=1)
    filtered["passes_filters"] = [passed for passed, _failures in outcomes]
    filtered["filter_failures"] = [failures for _passed, failures in outcomes]
    sort_cols = [col for col in ["passes_filters", "analysis_score"] if col in filtered.columns]
    if sort_cols:
        filtered = filtered.sort_values(sort_cols, ascending=[False, False][: len(sort_cols)]).reset_index(drop=True)
    filtered.insert(0, "display_rank", range(1, len(filtered) + 1))
    return filtered.head(max(1, int(keep_top_n)))


def _show_ranked_analysis_csv(
    path: Path,
    title: str = "App Re-analysis Results",
    thresholds: dict[str, float] | None = None,
    keep_top_n: int = 100,
) -> None:
    df = pd.read_csv(path)
    if thresholds is not None:
        df = _apply_live_filters(df, thresholds, keep_top_n)
    st.subheader(title)
    if df.empty:
        st.info("The ranked candidates table is empty.")
        return
    summary_cols = st.columns(4)
    summary_cols[0].metric("Candidates", len(df))
    if "passes_filters" in df:
        summary_cols[1].metric("Passing", int(df["passes_filters"].fillna(False).sum()))
    if "analysis_score" in df:
        summary_cols[2].metric("Best score", f"{pd.to_numeric(df['analysis_score'], errors='coerce').max():.2f}")
    if "ipsae_error" in df:
        summary_cols[3].metric("IPSAE errors", int(df["ipsae_error"].fillna("").astype(bool).sum()))
    visible_cols = [
        "display_rank",
        "analysis_rank",
        "candidate_id",
        "passes_filters",
        "native_pass_filters",
        "final_rank",
        "pass_filters",
        "quality_score",
        "analysis_score",
        "interaction_pae",
        "min_interaction_pae",
        "design_to_target_iptm",
        "design_ptm",
        "filter_rmsd",
        "designfolding-filter_rmsd",
        "native_rmsd",
        "native_rmsd_refolded",
        "binder_plddt",
        "confidence",
        "iptm",
        "ipsae",
        "ipsae_min",
        "ipsae_max",
        "ipsae_avg",
        "lis",
        "ipsae_min_in_calculation",
        "ipae",
        "pdockq_min",
        "pdockq_max",
        "pdockq2_min",
        "pdockq2_max",
        "ipde",
        "monomer_rmsd",
        "binder_rmsd",
        "hotspot_contact_fraction",
        "hotspots_contacted",
        "hotspot_contact_pairs",
        "min_binder_to_hotspot_distance",
        "binder_interface_contacts",
        "hotspot_interface_contact_fraction",
        "binder_radius_of_gyration",
        "binder_end_to_end_distance",
        "bindcraft_Target_RMSD",
        "bindcraft_Binder_Energy_Score",
        "bindcraft_ShapeComplementarity",
        "bindcraft_PackStat",
        "bindcraft_dG",
        "bindcraft_dSASA",
        "bindcraft_dG/dSASA",
        "bindcraft_Relaxed_Clashes",
        "bindcraft_n_InterfaceResidues",
        "bindcraft_n_InterfaceHbonds",
        "bindcraft_n_InterfaceUnsatHbonds",
        "bindcraft_Interface_SASA_%",
        "bindcraft_Interface_Hydrophobicity",
        "bindcraft_Binder_Helix%",
        "bindcraft_Binder_BetaSheet%",
        "bindcraft_Binder_Loop%",
        "bindcraft_TrajectoryTime",
        "filter_failures",
        "ipsae_error",
        "complex_pdb",
    ]
    st.dataframe(df[[col for col in visible_cols if col in df.columns]], use_container_width=True, hide_index=True)
    numeric_cols = [
        col
        for col in [
            "analysis_score",
            "ipsae",
            "ipsae_min",
            "lis",
            "ipae",
            "monomer_rmsd",
            "binder_rmsd",
            "hotspot_contact_fraction",
            "min_binder_to_hotspot_distance",
            "binder_radius_of_gyration",
            "binder_end_to_end_distance",
            "binder_plddt",
            "iptm",
        ]
        if col in df and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]
    if "analysis_rank" in df and "analysis_score" in df:
        st.caption("Score by rank")
        st.line_chart(df.set_index("analysis_rank")[["analysis_score"]])
    if len(numeric_cols) >= 2:
        scatter_cols = st.columns(2)
        with scatter_cols[0]:
            x_col = st.selectbox(f"{title} scatter X", numeric_cols, index=0, key=f"{title}_scatter_x")
        with scatter_cols[1]:
            y_col = st.selectbox(f"{title} scatter Y", numeric_cols, index=min(1, len(numeric_cols) - 1), key=f"{title}_scatter_y")
        st.scatter_chart(df, x=x_col, y=y_col, color="passes_filters" if "passes_filters" in df else None)
    _show_selected_design_structure(path, df, title)


def _source_is_selected_or_downstream(analysis_source: Path, selected_source: Path) -> bool:
    try:
        selected = selected_source.resolve()
        current = analysis_source.resolve()
    except OSError:
        return str(analysis_source) == str(selected_source)
    if current == selected:
        return True

    runs_root = selected.parent.parent
    for _step in range(20):
        metadata = read_json(current / "metadata.json")
        upstream_group = str(metadata.get("upstream_task_group") or "")
        upstream_run_id = str(metadata.get("upstream_run_id") or "")
        if not upstream_group or not upstream_run_id:
            return False
        current = runs_root / upstream_group / upstream_run_id
        try:
            if current.resolve() == selected:
                return True
        except OSError:
            return False
    return False


def _completed_analysis_runs(source_run_dir: Path) -> list[Path]:
    runs: list[Path] = []
    selected_source = source_run_dir.resolve()
    for job in collect_jobs("analysis"):
        run_dir = Path(str(job.get("run_dir") or ""))
        ranked_csv = run_dir / "artifacts" / "analysis" / "ranked_candidates.csv"
        if not ranked_csv.exists():
            continue
        input_payload = read_json(run_dir / "input.json")
        analysis_source = Path(str((input_payload.get("inputs") or {}).get("source_run_dir") or ""))
        if not _source_is_selected_or_downstream(analysis_source, selected_source):
            continue
        runs.append(run_dir)
    runs.sort(key=lambda run_dir: read_json(run_dir / "metadata.json").get("created_at") or run_dir.name, reverse=True)
    return runs


def _analysis_result_label(run_dir: Path) -> str:
    metadata = read_json(run_dir / "metadata.json")
    result = read_json(run_dir / "result.json")
    metrics = result.get("metrics") or {}
    created = str(metadata.get("created_at") or "").replace("T", " ").split("+", 1)[0]
    parts = [str(metadata.get("job_code") or run_dir.name[-5:])]
    if created:
        parts.append(created)
    upstream = metadata.get("upstream_job_code")
    if upstream:
        parts.append(f"from {upstream}")
    candidate_count = metrics.get("candidate_count")
    passing_count = metrics.get("passing_count")
    if candidate_count is not None:
        parts.append(f"{candidate_count} candidates")
    if passing_count is not None:
        parts.append(f"{passing_count} passing")
    return " | ".join(parts)


completed_analysis_runs = _completed_analysis_runs(Path(str(source["run_dir"])))

st.subheader("Completed App Re-analysis Jobs")
completed_run = None
if completed_analysis_runs:
    selected_completed = st.selectbox(
        "Re-analysis job",
        range(len(completed_analysis_runs)),
        format_func=lambda index: _analysis_result_label(completed_analysis_runs[index]),
    )
    completed_run = completed_analysis_runs[selected_completed]
    st.link_button("Open selected analysis job", result_link("analysis", completed_run.name))
else:
    st.info("No app re-analysis jobs for this selected candidate set yet.")

st.subheader("App Re-analysis Filters")
tool = st.segmented_control(
    "Analysis mode",
    ["ranking", "filters", "reports"],
    selection_mode="single",
    default="ranking",
    format_func={
        "ranking": "Ranking",
        "filters": "Filters",
        "reports": "Reports",
    }.get,
)
keep_top_n = st.number_input("Keep top candidates", min_value=1, max_value=10000, value=100, step=10)
cols = st.columns(3)
with cols[0]:
    min_binder_plddt = st.number_input("Min binder pLDDT", min_value=0.0, max_value=100.0, value=DEFAULT_THRESHOLDS["min_binder_plddt"], step=1.0)
    max_ipae = st.number_input("Max iPAE", min_value=0.0, max_value=100.0, value=DEFAULT_THRESHOLDS["max_ipae"], step=1.0)
with cols[1]:
    min_iptm = st.number_input("Min ipTM", min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLDS["min_iptm"], step=0.05)
    max_ipde = st.number_input("Max iPDE", min_value=0.0, max_value=100.0, value=DEFAULT_THRESHOLDS["max_ipde"], step=1.0)
with cols[2]:
    min_ipsae = st.number_input("Min ipSAE", min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLDS["min_ipsae"], step=0.05)
    min_confidence = st.number_input("Min confidence", min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLDS["min_confidence"], step=0.05)
max_binder_rmsd = st.number_input("Max binder RMSD", min_value=0.0, max_value=100.0, value=DEFAULT_THRESHOLDS["max_binder_rmsd"], step=0.5)
hotspot_filter_enabled = st.checkbox("Filter by hotspot/site recovery", value=False)
hotspot_cols = st.columns(2)
with hotspot_cols[0]:
    min_hotspot_contact_fraction = st.number_input(
        "Min hotspot contact fraction",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        disabled=not hotspot_filter_enabled,
    )
with hotspot_cols[1]:
    max_hotspot_distance = st.number_input(
        "Max nearest hotspot distance",
        min_value=0.0,
        max_value=50.0,
        value=8.0,
        step=0.5,
        disabled=not hotspot_filter_enabled,
    )

thresholds = {
    "min_binder_plddt": float(min_binder_plddt),
    "min_confidence": float(min_confidence),
    "min_iptm": float(min_iptm),
    "min_ipsae": float(min_ipsae),
    "max_ipae": float(max_ipae),
    "max_ipde": float(max_ipde),
    "max_binder_rmsd": float(max_binder_rmsd),
}
if hotspot_filter_enabled:
    thresholds["min_hotspot_contact_fraction"] = float(min_hotspot_contact_fraction)
    thresholds["max_hotspot_distance"] = float(max_hotspot_distance)

if completed_run is not None:
    _show_ranked_analysis_csv(
        completed_run / "artifacts" / "analysis" / "ranked_candidates.csv",
        thresholds=thresholds,
        keep_top_n=int(keep_top_n),
    )
else:
    st.info("Run app re-analysis to create a normalized ranking table.")

if source_tool not in native_end_to_end_tools:
    st.subheader("Run App Re-analysis")

if source_tool not in native_end_to_end_tools and st.button("Run app re-analysis", type="primary"):
    try:
        run_dir = run_analysis_contract(
            source_run_dir=source["run_dir"],
            candidates_jsonl=source["candidates_jsonl"],
            tool=str(tool or "ranking"),
            keep_top_n=int(keep_top_n),
            thresholds=thresholds,
        )
        st.success("App re-analysis job finished.")
        st.link_button("Open result", result_link("analysis", run_dir.name))
        ranked_csv = run_dir / "artifacts" / "analysis" / "ranked_candidates.csv"
        if ranked_csv.exists():
            _show_ranked_analysis_csv(
                ranked_csv,
                title="New Completed Analysis Results",
                thresholds=thresholds,
                keep_top_n=int(keep_top_n),
            )
    except Exception as exc:
        st.error(str(exc))
