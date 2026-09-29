from __future__ import annotations

import ast
import gzip
import hashlib
import json
import shlex
from io import StringIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from mn_protein_design.app.pages.common import (
    esmfold2_preset_label,
    esmfold2_preset_selector,
    cpu_run_panel,
    gpu_run_panel,
    refresh_results_button,
    result_link,
    selected_dataframe_rows,
    show_delete_jobs_dialog,
    show_pipeline_links,
    source_from_run_dir,
)
from mn_protein_design.core.benchmark_presets import load_feature_presets
from mn_protein_design.core.candidates import (
    STAGE_COMPLEX_REFOLDING,
    STAGE_GENERATION_BACKBONE_SEQUENCE,
    STAGE_SEQUENCE_DESIGN,
)
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.core.jobs import ACTIVE_STATUSES, collect_jobs, read_json
from mn_protein_design.core.workflow_queue import queued_workflow_alias
from mn_protein_design.workflows.campaigns import run_validation_sequence
from mn_protein_design.core.runtime_estimator import estimate_engines
from mn_protein_design.workflows import benchmark as benchmark_workflow
from mn_protein_design.workflows import refolding as refolding_workflow
from mn_protein_design.workflows.benchmark import (
    ALPHAFAST_DB_DIR,
    ALPHAFAST_WEIGHTS_DIR,
    COLABFOLD_CACHE_DIR,
    MSA_REPOSITORY_DIR,
)
from mn_protein_design.workflows.capacity_benchmark import capacity_warnings
from mn_protein_design.workflows.esm_binder import ESMFOLD2_MODEL_DIR
from mn_protein_design.workflows.refolding import BOLTZ_MODELS_DIR


queue_esmfold2_complex_validation = queued_workflow_alias(
    refolding_workflow.run_esmfold2_complex_validation,
    task_group="refolding-validation",
    tool="esmfold2_complex_validation",
    job_type="complex_refolding",
)
queue_validation_sequence = queued_workflow_alias(
    run_validation_sequence,
    task_group="refolding-validation",
    tool="full_validation_pipeline",
    job_type="validation_sequence",
)
from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates


KNOWN_BENCHMARK_ROOT = Path("/mnt/db/reference_files/de_novo_binder_scoring_overath_2025")
KNOWN_BENCHMARK_CSV = KNOWN_BENCHMARK_ROOT / "final_dataset.csv"
KNOWN_BENCHMARK_PDB_DIR = KNOWN_BENCHMARK_ROOT / "input_pdbs"

STRUCTURE_ENGINE_TABLES = [
    ("AF3", "alphafast_af3", "alphafast_af3_metrics.csv"),
    ("AF2-IG", "af2_initial_guess", "af2_initial_guess_metrics.csv"),
    ("Boltz-2", "boltz2", "boltz2_initial_guess_metrics.csv"),
    ("BoltzGen Fold", "boltzgen_fold", "boltzgen_fold_metrics.csv"),
    ("ColabFold", "colabfold", "colabfold_metrics.csv"),
    ("ESMFold2", "esmfold2", "esmfold2_metrics.csv"),
    ("OpenFold-3", "openfold3", "openfold3_metrics.csv"),
    ("Protenix", "protenix", "protenix_metrics.csv"),
    ("Protenix v1", "protenix_v1", "protenix_v1_metrics.csv"),
    ("Protenix v2", "protenix_v2", "protenix_v2_metrics.csv"),
    ("RF3", "rf3", "rf3_metrics.csv"),
]
STRUCTURE_ENGINE_BY_LABEL = {label: (engine_key, csv_name) for label, engine_key, csv_name in STRUCTURE_ENGINE_TABLES}
STRUCTURE_ENGINE_BY_LABEL.update(
    {
        "ESMFold2 Fast": ("esmfold2", "esmfold2_metrics.csv"),
        "ESMFold2 Standard": ("esmfold2", "esmfold2_metrics.csv"),
        "ESMFold2 Careful": ("esmfold2", "esmfold2_metrics.csv"),
        "ESMFold2 High diffusion": ("esmfold2", "esmfold2_metrics.csv"),
        "ESMFold2 Design rank": ("esmfold2", "esmfold2_metrics.csv"),
        "ESMFold2 Custom": ("esmfold2", "esmfold2_metrics.csv"),
    }
)
STRUCTURE_ENGINE_METRIC_PDB_KEYS = {
    "alphafast_af3": "af3",
    "af2_initial_guess": "af2",
    "boltz2": "boltz2",
    "boltzgen_fold": "boltzgen_fold",
    "colabfold": "colab",
    "esmfold2": "esmfold2",
    "openfold3": "openfold3",
    "protenix": "protenix",
    "protenix_v1": "protenix_v1",
    "protenix_v2": "protenix_v2",
    "rf3": "rf3",
}


st.title("Refolding / Validation")
st.caption("Consumes normalized candidates and runs refolding/validation engines on existing designs.")

sources = candidate_sources()
usable_sources = [
    source
    for source in sources
    if any(
        stage in source["stage_counts"]
        for stage in [STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_COMPLEX_REFOLDING]
    )
]

if not usable_sources:
    st.info("No sequence-designed or accepted-complex candidate sets are available yet. Run a generation tool first.")
    st.stop()

labels = [
    f"{source['job_code']} | {source['tool']} | {source['candidate_count']} candidates | {source['stage_counts']}"
    for source in usable_sources
]

def _safe_sort_token(value: object) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "sort"))
    return token.strip("_") or "sort"


def _parse_chain_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip().strip("'\"") for part in text.replace(";", ",").split(",") if part.strip().strip("'\"")]


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


def _candidate_total_length(candidate: dict) -> int | None:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    for key in ("total_length", "total_residues", "capacity_total_length"):
        try:
            value = metrics.get(key) if key in metrics else candidate.get(key)
            if value is not None and str(value).strip():
                return int(float(str(value)))
        except (TypeError, ValueError):
            pass
    try:
        binder_length = int(float(str(candidate.get("binder_length") or metrics.get("binder_length") or "")))
        target_length = int(float(str(metrics.get("target_length") or candidate.get("target_length") or "")))
        if binder_length > 0 and target_length > 0:
            return binder_length + target_length
    except (TypeError, ValueError):
        pass
    sequence = str(candidate.get("binder_sequence") or "")
    return len(sequence) if sequence else None


def _show_capacity_warnings(warnings: list[dict[str, object]]) -> None:
    relevant = [row for row in warnings if row.get("severity") in {"error", "warning"}]
    unknown = [row for row in warnings if row.get("severity") == "info"]
    if relevant:
        st.warning("Capacity benchmark warning: at least one selected engine is outside known successful limits.")
        st.dataframe(
            pd.DataFrame(relevant),
            hide_index=True,
            width="stretch",
            column_order=[
                "engine",
                "severity",
                "evidence_scope",
                "message",
                "max_success_total_length",
                "min_failed_total_length",
                "min_oom_total_length",
                "preset",
                "gpu_device",
            ],
        )
    elif warnings and all(row.get("severity") == "ok" for row in warnings):
        st.success("Selected system size is within the matching capacity benchmark range.")
    if unknown:
        with st.expander("Capacity evidence not available for some engines", expanded=False):
            st.dataframe(pd.DataFrame(unknown), hide_index=True, width="stretch")


def _resolve_source_path(source: dict, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(str(source["run_dir"])) / path
    return path


@st.cache_data(show_spinner=False)
def _structure_chain_options(path_text: str) -> list[str]:
    path = Path(path_text).expanduser()
    if not path.exists():
        return []
    try:
        return refolding_workflow._structure_chains(path)
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _benchmark_target_library() -> list[dict[str, object]]:
    if not KNOWN_BENCHMARK_CSV.exists() or not KNOWN_BENCHMARK_PDB_DIR.exists():
        return []
    try:
        columns = ["binder_id", "target_id", "source", "target_chains"]
        dataset = pd.read_csv(KNOWN_BENCHMARK_CSV, usecols=columns)
    except Exception:
        return []
    rows: list[dict[str, object]] = []
    for (target_id, source_name), group in dataset.groupby(["target_id", "source"], dropna=False, sort=True):
        representative = group.iloc[0]
        binder_id = str(representative.get("binder_id") or "").strip()
        pdb_path = KNOWN_BENCHMARK_PDB_DIR / f"{binder_id}.pdb"
        if not binder_id or not pdb_path.exists():
            continue
        chains = _parse_chain_list(representative.get("target_chains"))
        if not chains:
            chains = _structure_chain_options(str(pdb_path))
        rows.append(
            {
                "label": f"Overath 2025 benchmark | {target_id} | {source_name}",
                "target_id": str(target_id),
                "source": str(source_name),
                "origin": "Installed Overath 2025 benchmark representative input complex",
                "path": str(pdb_path),
                "chains": chains,
                "records": int(len(group)),
            }
        )
    return rows


def _source_target_library(source: dict, candidates: list[dict]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        target_path = _resolve_source_path(source, candidate.get("target_pdb"))
        if target_path is None or not target_path.exists():
            continue
        target_id = str(candidate.get("target_id") or target_path.stem)
        target_chains = [str(chain) for chain in candidate.get("target_chains") or []]
        if not target_chains:
            target_chains = _structure_chain_options(str(target_path))
        source_tool = str(candidate.get("source_tool") or source.get("tool") or "candidate set")
        key = (target_id, str(target_path), tuple(target_chains))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "label": f"{source['job_code']} candidate target | {target_id}",
                "target_id": target_id,
                "source": source_tool,
                "origin": f"Selected/imported candidate set {source['job_code']}",
                "path": str(target_path),
                "chains": target_chains,
                "records": sum(
                    1
                    for item in candidates
                    if str(item.get("target_id") or target_path.stem) == target_id
                    and _resolve_source_path(source, item.get("target_pdb")) == target_path
                ),
            }
        )
    return rows


def _short_path(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).name


def _first_run_csv_row(run_dir: Path) -> dict[str, object]:
    run_csv = run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv"
    if not run_csv.exists():
        return {}
    try:
        frame = pd.read_csv(run_csv, nrows=1)
    except Exception:
        return {}
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _refolding_target_summary(run_dir: Path, input_json: dict) -> dict[str, object]:
    inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
    row = _first_run_csv_row(run_dir)
    target_pdb = row.get("source_target_pdb") or row.get("target_pdb") or inputs.get("target_override_pdb")
    target_chains = _parse_chain_list(row.get("target_chains") or inputs.get("target_override_chains"))
    source_target_chains = _parse_chain_list(row.get("source_target_chains"))
    target_source = str(row.get("target_source") or ("override" if inputs.get("target_override_pdb") else "candidate"))
    target_id = str(row.get("target_id") or Path(str(target_pdb or "target")).stem)
    target_len = None
    for key, value in row.items():
        if str(key).startswith("target_subchain_") and str(key).endswith("_len"):
            target_len = value
            break
    return {
        "target_id": target_id,
        "target_source": target_source,
        "target_chains": ",".join(target_chains),
        "source_target_chains": ",".join(source_target_chains),
        "target_pdb": _short_path(target_pdb),
        "target_length": target_len,
    }


def _refolding_result_rows() -> list[dict[str, object]]:
    rows = []

    def _first_positive_int(*values: object) -> int | None:
        for value in values:
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(number) and int(number) > 0:
                return int(number)
        return None

    for job in collect_jobs("benchmark"):
        run_dir = Path(job["run_dir"])
        input_json = read_json(run_dir / "input.json")
        if str(input_json.get("job_type") or "") != "refolding_evaluation":
            continue
        inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
        params = input_json.get("params") if isinstance(input_json.get("params"), dict) else {}
        result = read_json(run_dir / "result.json")
        result_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        worker_payload = read_json(run_dir / "worker_request.json")
        worker_kwargs = worker_payload.get("kwargs") if isinstance(worker_payload.get("kwargs"), dict) else {}
        esm_loops = params.get("num_loops", worker_kwargs.get("num_loops"))
        esm_steps = params.get("num_sampling_steps", worker_kwargs.get("num_sampling_steps"))
        target_summary = _refolding_target_summary(run_dir, input_json)
        selected_candidate_ids = inputs.get("selected_candidate_ids") if isinstance(inputs.get("selected_candidate_ids"), list) else []
        candidate_count = _first_positive_int(
            len(selected_candidate_ids) if selected_candidate_ids else None,
            params.get("selected_candidate_count"),
            params.get("candidate_count"),
            params.get("staged_count"),
            result_metrics.get("record_count"),
            result_metrics.get("staged_metadata_rows"),
            result_metrics.get("chain_msa_record_count"),
        )
        source_run_dir_text = str(inputs.get("source_run_dir") or "").strip()
        source_run_dir = Path(source_run_dir_text) if source_run_dir_text else None
        source_run_id = source_run_dir.name if source_run_dir is not None else ""
        source_metadata = read_json(source_run_dir / "metadata.json") if source_run_dir is not None else {}
        source_job_code = str(source_metadata.get("job_code") or "").strip()
        source_task_group = str(source_metadata.get("task_group") or (source_run_dir.parent.name if source_run_dir is not None else "")).strip()
        source_job = _short_path(source_run_dir_text)
        candidate_signature = hashlib.sha1(
            "\n".join(sorted(str(candidate_id).lower() for candidate_id in selected_candidate_ids)).encode("utf-8")
        ).hexdigest()[:8] if selected_candidate_ids else "unknown"
        design_set = f"{source_job or 'unknown source'} | {candidate_count or 'unknown'} designs | {candidate_signature}"
        rows.append(
            {
                "result": result_link("benchmark", str(job.get("run_id"))),
                "description": job.get("description"),
                "status": job.get("status"),
                "design_set": design_set,
                "design_set_hash": candidate_signature,
                "source_job": source_job,
                "source_job_code": source_job_code,
                "source_run_id": source_run_id,
                "source_task_group": source_task_group,
                "candidates": candidate_count,
                **target_summary,
                "esmfold2_preset": esmfold2_preset_label(esm_loops, esm_steps),
                "esmfold2_loops": esm_loops,
                "esmfold2_steps": esm_steps,
                "records": result_metrics.get("record_count"),
                "current_phase": job.get("current_phase"),
                "current_engine": job.get("current_engine"),
                "created_at": job.get("created_at"),
                "job_code": job.get("job_code"),
                "run_id": job.get("run_id"),
                "task_group": "benchmark",
                "run_dir": str(run_dir),
            }
        )
    return rows


def _metrics_csv_path(run_dir: Path) -> Path:
    return run_dir / "artifacts" / "benchmark" / "merged_benchmark_metrics.csv"


def _candidate_input_rank_map(run_dir: Path) -> dict[str, int]:
    input_json = read_json(run_dir / "input.json")
    inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
    selected_candidate_ids = inputs.get("selected_candidate_ids") if isinstance(inputs.get("selected_candidate_ids"), list) else []
    rank_map = {str(candidate_id).lower(): rank for rank, candidate_id in enumerate(selected_candidate_ids, start=1)}
    candidates_jsonl = Path(str(inputs.get("candidates_jsonl") or ""))
    if candidates_jsonl.exists():
        try:
            with candidates_jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    candidate_id = str(row.get("candidate_id") or "").lower()
                    if not candidate_id:
                        continue
                    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                    rank_value = (
                        metrics.get("bindcraft_final_rank")
                        or metrics.get("bindcraft_Rank")
                        or metrics.get("import_rank")
                    )
                    rank_number = pd.to_numeric(pd.Series([rank_value]), errors="coerce").iloc[0]
                    if pd.notna(rank_number):
                        rank_map[candidate_id] = int(rank_number)
        except Exception:
            pass
    return rank_map


def _rank_ascending(feature: str) -> bool:
    text = feature.lower()
    lower_is_better = ["pae", "ipae", "rmsd", "ddg", "dg", "clash", "error", "distance"]
    return any(token in text for token in lower_is_better)


def _feature_engine_name(feature: str) -> str:
    text = str(feature or "").lower()
    if text.startswith(("alphafast_af3_", "af3_")):
        return "AF3"
    if text.startswith("af2_"):
        return "AF2-IG"
    if text.startswith("boltz2_"):
        return "Boltz-2"
    if text.startswith("boltzgen_fold_"):
        return "BoltzGen Fold"
    if text.startswith("colab_"):
        return "ColabFold"
    if text.startswith("esmfold2_fast_"):
        return "ESMFold2 Fast"
    if text.startswith("esmfold2_standard_"):
        return "ESMFold2 Standard"
    if text.startswith("esmfold2_careful_"):
        return "ESMFold2 Careful"
    if text.startswith("esmfold2_high_diffusion_"):
        return "ESMFold2 High diffusion"
    if text.startswith("esmfold2_design_rank_"):
        return "ESMFold2 Design rank"
    if text.startswith("esmfold2_custom_"):
        return "ESMFold2 Custom"
    if text.startswith("esmfold2_"):
        return "ESMFold2"
    if text.startswith("protenix_"):
        return "Protenix"
    if text.startswith("rf3_"):
        return "RF3"
    if text.startswith("input_") or text.startswith("bindcraft_"):
        return "Input / BindCraft"
    return "Other"


def _engine_sort_key(engine: str) -> tuple[int, str]:
    order = [
        "Input / BindCraft",
        "AF3",
        "AF2-IG",
        "Boltz-2",
        "BoltzGen Fold",
        "ColabFold",
        "ESMFold2",
        "ESMFold2 Fast",
        "ESMFold2 Standard",
        "ESMFold2 Careful",
        "ESMFold2 High diffusion",
        "ESMFold2 Design rank",
        "ESMFold2 Custom",
        "Protenix",
        "RF3",
        "Other",
    ]
    try:
        return (order.index(engine), engine)
    except ValueError:
        return (len(order), engine)


def _preset_features_by_engine(preset: dict, numeric_cols: list[str]) -> dict[str, str]:
    matched: dict[str, str] = {}
    available = set(numeric_cols)
    rows = preset.get("features") if isinstance(preset.get("features"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature") or "")
        if feature not in available:
            continue
        engine = _feature_engine_name(feature)
        if engine == "Other" and row.get("engine"):
            engine = str(row.get("engine"))
        if engine not in matched:
            matched[engine] = feature
    return matched


def _preset_directions_by_feature(preset: dict) -> dict[str, str]:
    directions: dict[str, str] = {}
    rows = preset.get("features") if isinstance(preset.get("features"), list) else []
    for row in rows:
        if isinstance(row, dict) and row.get("feature"):
            directions[str(row.get("feature"))] = str(row.get("direction") or "").lower()
    return directions


def _numeric_compare_columns(frames: list[pd.DataFrame]) -> list[str]:
    if not frames:
        return []
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    ignored = {"binder_id", "target_id", "target_source", "label", "source"}
    numeric_cols = []
    for column in sorted(common):
        if column in ignored:
            continue
        if all(pd.to_numeric(frame[column], errors="coerce").notna().any() for frame in frames):
            numeric_cols.append(column)
    return numeric_cols


def _remap_moved_run_path(run_dir: Path, path: Path) -> Path | None:
    if not path.is_absolute():
        return None
    parts = path.parts
    if run_dir.name in parts:
        index = parts.index(run_dir.name)
        candidate = run_dir / Path(*parts[index + 1 :])
        if candidate.exists():
            return candidate
    if "artifacts" in parts:
        index = parts.index("artifacts")
        candidate = run_dir / "artifacts" / Path(*parts[index + 1 :])
        if candidate.exists():
            return candidate
    return None


def _resolve_run_relative_path(run_dir: Path, value: object) -> Path | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path] if path.is_absolute() else [run_dir / path]
    remapped = _remap_moved_run_path(run_dir, path)
    if remapped is not None:
        candidates.insert(0, remapped)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _viewer_structure_path(run_dir: Path, engine_key: str, design_id: str) -> Path | None:
    metric_key = STRUCTURE_ENGINE_METRIC_PDB_KEYS.get(engine_key)
    if not metric_key:
        return None
    path = run_dir / "artifacts" / "benchmark" / "predicted_viewer_structures" / metric_key / f"{design_id}.pdb"
    return path if path.exists() else None


def _metric_structure_path(run_dir: Path, engine_key: str, design_id: str) -> Path | None:
    metric_key = STRUCTURE_ENGINE_METRIC_PDB_KEYS.get(engine_key)
    if not metric_key:
        return None
    path = run_dir / "artifacts" / "benchmark" / "predicted_metric_pdbs" / metric_key / f"{design_id}.pdb"
    return path if path.exists() else None


def _resolve_engine_structure_path(run_dir: Path, engine_key: str, value: object) -> Path | None:
    path = _resolve_run_relative_path(run_dir, value)
    if path is not None and path.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
        return path
    text = str(value or "").strip()
    if text and not Path(text).is_absolute():
        candidate = run_dir / "artifacts" / "engines" / engine_key / text
        if candidate.exists() and candidate.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
            return candidate
    return None


def _alphafast_model_from_summary(run_dir: Path, value: object) -> Path | None:
    summary_path = _resolve_run_relative_path(run_dir, value)
    if summary_path is None:
        return None
    model_path = summary_path.with_name(summary_path.name.replace("_summary_confidences.json", "_model.cif"))
    return model_path if model_path.exists() else None


def _colabfold_model_for_design(run_dir: Path, design_id: str) -> Path | None:
    for root in [
        run_dir / "artifacts" / "benchmark" / "predicted_viewer_structures" / "colab",
        run_dir / "artifacts" / "engines" / "colabfold",
        run_dir / "artifacts" / "raw" / "colabfold",
        run_dir / "artifacts" / "benchmark" / "predicted_metric_pdbs" / "colab",
    ]:
        if not root.exists():
            continue
        matches = sorted(root.rglob(f"*{design_id}*.pdb")) + sorted(root.rglob(f"*{design_id}*.cif"))
        if matches:
            return matches[0]
    return None


def _engine_structure_for_design(run_dir: Path, engine_label: str, design_id: str) -> tuple[Path | None, str]:
    engine_key, csv_name = STRUCTURE_ENGINE_BY_LABEL.get(engine_label, (None, None))
    if not engine_key or not csv_name:
        return None, "unsupported engine"
    benchmark_dir = run_dir / "artifacts" / "benchmark"
    path = _viewer_structure_path(run_dir, engine_key, design_id) or _metric_structure_path(run_dir, engine_key, design_id)
    table_path = benchmark_dir / csv_name
    if table_path.exists():
        try:
            table = pd.read_csv(table_path)
        except Exception:
            table = pd.DataFrame()
        if not table.empty and "binder_id" in table.columns:
            subset = table[table["binder_id"].astype(str).str.lower().eq(str(design_id).lower())]
            if not subset.empty:
                row = subset.iloc[0]
                if path is None and "complex_pdb" in subset.columns:
                    path = _resolve_engine_structure_path(run_dir, engine_key, row.get("complex_pdb"))
                if path is None and engine_key == "alphafast_af3" and "summary_confidences" in subset.columns:
                    path = _alphafast_model_from_summary(run_dir, row.get("summary_confidences"))
                if path is None and engine_key == "colabfold":
                    path = _colabfold_model_for_design(run_dir, design_id)
    if path is None:
        return None, "structure not found"
    return path, "ok"


def _mmcif_text_with_default_occupancy(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    atom_site_headers: list[str] = []
    in_atom_site_loop = False
    inserted_header = False
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_site_loop = True
            atom_site_headers = []
            inserted_header = False
            output.append(line)
            continue
        if in_atom_site_loop and stripped.startswith("_atom_site."):
            if stripped not in atom_site_headers:
                atom_site_headers.append(stripped)
            output.append(line)
            if stripped == "_atom_site.B_iso_or_equiv" and "_atom_site.occupancy" not in atom_site_headers:
                output.append("_atom_site.occupancy")
                atom_site_headers.append("_atom_site.occupancy")
                inserted_header = True
            continue
        if in_atom_site_loop and stripped.startswith(("ATOM ", "HETATM ")):
            tokens = shlex.split(stripped, posix=False)
            if inserted_header and len(tokens) == len(atom_site_headers) - 1:
                tokens.insert(atom_site_headers.index("_atom_site.occupancy"), "1.00")
                output.append(" ".join(tokens))
            else:
                output.append(line)
            continue
        if in_atom_site_loop and stripped.startswith("#"):
            in_atom_site_loop = False
            atom_site_headers = []
            inserted_header = False
        output.append(line)
    return "\n".join(output) + "\n"


def _structure_path_is_cif(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz"))


def _read_structure_text(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            magic = handle.read(2)
    except OSError:
        magic = b""
    if path.name.lower().endswith(".gz") or magic == b"\x1f\x8b":
        with gzip.open(path, "rt", errors="ignore") as handle:
            return handle.read()
    return path.read_text(errors="ignore")


def _parse_structure_file(path: Path):
    from Bio.PDB import MMCIFParser, PDBParser

    text = _read_structure_text(path)
    if _structure_path_is_cif(path):
        parser = MMCIFParser(QUIET=True)
        try:
            return parser.get_structure(path.stem, StringIO(text))
        except KeyError as exc:
            if str(exc).strip("'\"") != "_atom_site.occupancy":
                raise
            patched_text = _mmcif_text_with_default_occupancy(text)
            return parser.get_structure(path.stem, StringIO(patched_text))
    return PDBParser(QUIET=True).get_structure(path.stem, StringIO(text))


def _target_ca_atoms(structure: object, target_chains: list[str]) -> list[object]:
    model = next(structure.get_models(), None)
    if model is None:
        return []
    atoms: list[object] = []
    for chain_id in target_chains:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            if "CA" in residue:
                atoms.append(residue["CA"])
    return atoms


def _structure_to_pdb_text(structure: object) -> str:
    from Bio.PDB import PDBIO

    output = StringIO()
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(output)
    return output.getvalue()


def _aligned_structure_text(path: Path, fixed_structure: object, fixed_target_chains: list[str], moving_target_chains: list[str]) -> tuple[str | None, float | None, str]:
    from Bio.PDB import Superimposer

    try:
        moving_structure = _parse_structure_file(path)
        fixed_atoms = _target_ca_atoms(fixed_structure, fixed_target_chains)
        moving_atoms = _target_ca_atoms(moving_structure, moving_target_chains)
    except Exception as exc:
        return None, None, f"alignment failed: {exc}"
    atom_count = min(len(fixed_atoms), len(moving_atoms))
    if atom_count < 3:
        return None, None, f"alignment skipped: only {atom_count} shared target CA atoms"
    superimposer = Superimposer()
    superimposer.set_atoms(fixed_atoms[:atom_count], moving_atoms[:atom_count])
    superimposer.apply(moving_structure.get_atoms())
    return (
        _structure_to_pdb_text(moving_structure),
        float(superimposer.rms),
        f"aligned on {atom_count} target CAs ({','.join(moving_target_chains)} -> {','.join(fixed_target_chains)})",
    )


def _pdb_text_to_structure(text: str, structure_id: str):
    from Bio.PDB import PDBParser

    return PDBParser(QUIET=True).get_structure(structure_id, StringIO(text))


def _residue_key(residue: object) -> tuple[str, int, str]:
    residue_id = residue.id
    return (str(residue.get_parent().id), int(residue_id[1]), str(residue_id[2]).strip())


def _ca_by_residue(structure: object, chains: list[str]) -> dict[tuple[str, int, str], object]:
    model = next(structure.get_models(), None)
    if model is None:
        return {}
    wanted = set(chains)
    atoms: dict[tuple[str, int, str], object] = {}
    for chain in model:
        if chain.id not in wanted:
            continue
        for residue in chain:
            if residue.id[0] == " " and "CA" in residue:
                atoms[_residue_key(residue)] = residue["CA"]
    return atoms


def _heavy_atoms_for_chains(structure: object, chains: list[str]) -> list[object]:
    model = next(structure.get_models(), None)
    if model is None:
        return []
    wanted = set(chains)
    atoms: list[object] = []
    for chain in model:
        if chain.id not in wanted:
            continue
        for atom in chain.get_atoms():
            if str(getattr(atom, "element", "")).upper() == "H":
                continue
            atoms.append(atom)
    return atoms


def _contact_target_residues(structure: object, binder_chains: list[str], target_chains: list[str], cutoff: float) -> set[tuple[str, int, str]]:
    from Bio.PDB import NeighborSearch

    binder_atoms = _heavy_atoms_for_chains(structure, binder_chains)
    target_atoms = _heavy_atoms_for_chains(structure, target_chains)
    if not binder_atoms or not target_atoms:
        return set()
    target_search = NeighborSearch(target_atoms)
    contacts: set[tuple[str, int, str]] = set()
    for binder_atom in binder_atoms:
        for target_atom in target_search.search(binder_atom.coord, float(cutoff), level="A"):
            contacts.add(_residue_key(target_atom.get_parent()))
    return contacts


def _atom_rmsd(atom_pairs: list[tuple[object, object]]) -> float | None:
    if not atom_pairs:
        return None
    total = 0.0
    for fixed_atom, moving_atom in atom_pairs:
        delta = fixed_atom.coord - moving_atom.coord
        total += float((delta * delta).sum())
    return (total / len(atom_pairs)) ** 0.5


def _input_geometry_for_design(run_a_dir: Path, run_b_dir: Path, design_id: str, contact_cutoff: float) -> dict[str, object]:
    from Bio.PDB import Superimposer

    row_a = _run_csv_row_for_design(run_a_dir, design_id)
    row_b = _run_csv_row_for_design(run_b_dir, design_id)
    complex_a = _resolve_run_relative_path(run_a_dir, row_a.get("complex_pdb"))
    complex_b = _resolve_run_relative_path(run_b_dir, row_b.get("complex_pdb"))
    if complex_a is None or complex_b is None:
        return {"binder_id": design_id, "geometry_status": "missing input complex"}
    target_chains_a = _parse_chain_list(row_a.get("target_chains")) or _run_target_chains(run_a_dir)
    target_chains_b = _parse_chain_list(row_b.get("target_chains")) or _run_target_chains(run_b_dir)
    binder_chains_a = _parse_chain_list(row_a.get("binder_chains")) or ["A"]
    binder_chains_b = _parse_chain_list(row_b.get("binder_chains")) or ["A"]
    try:
        structure_a = _parse_structure_file(complex_a)
        structure_b = _parse_structure_file(complex_b)
        target_atoms_a = _target_ca_atoms(structure_a, target_chains_a)
        target_atoms_b = _target_ca_atoms(structure_b, target_chains_b)
    except Exception as exc:
        return {"binder_id": design_id, "geometry_status": f"parse failed: {exc}"}
    atom_count = min(len(target_atoms_a), len(target_atoms_b))
    if atom_count < 3:
        return {"binder_id": design_id, "geometry_status": f"target alignment skipped: {atom_count} shared CAs"}
    superimposer = Superimposer()
    superimposer.set_atoms(target_atoms_a[:atom_count], target_atoms_b[:atom_count])
    superimposer.apply(structure_b.get_atoms())
    binder_atoms_a = _target_ca_atoms(structure_a, binder_chains_a)
    binder_atoms_b = _target_ca_atoms(structure_b, binder_chains_b)
    binder_count = min(len(binder_atoms_a), len(binder_atoms_b))
    binder_rmsd = _atom_rmsd(list(zip(binder_atoms_a[:binder_count], binder_atoms_b[:binder_count])))
    contacts_a = _contact_target_residues(structure_a, binder_chains_a, target_chains_a, contact_cutoff)
    contacts_b = _contact_target_residues(structure_b, binder_chains_b, target_chains_b, contact_cutoff)
    ca_a = _ca_by_residue(structure_a, target_chains_a)
    ca_b = _ca_by_residue(structure_b, target_chains_b)
    site_keys = sorted((contacts_a | contacts_b) & set(ca_a) & set(ca_b))
    site_rmsd = _atom_rmsd([(ca_a[key], ca_b[key]) for key in site_keys])
    return {
        "binder_id": design_id,
        "geometry_status": "ok",
        "target_ca_rmsd": float(superimposer.rms),
        "target_ca_count": atom_count,
        "binder_ca_rmsd": binder_rmsd,
        "binder_ca_count": binder_count,
        "binding_site_target_ca_rmsd": site_rmsd,
        "binding_site_ca_count": len(site_keys),
        "run_a_contact_residues": len(contacts_a),
        "run_b_contact_residues": len(contacts_b),
    }


@st.cache_data(show_spinner=False)
def _input_geometry_table(run_a_dir_text: str, run_b_dir_text: str, design_ids: tuple[str, ...], contact_cutoff: float) -> pd.DataFrame:
    run_a_dir = Path(run_a_dir_text)
    run_b_dir = Path(run_b_dir_text)
    rows = [
        _input_geometry_for_design(run_a_dir, run_b_dir, str(design_id), float(contact_cutoff))
        for design_id in design_ids
    ]
    return pd.DataFrame(rows)


def _prediction_geometry_for_design(run_a_dir: Path, run_b_dir: Path, engine_label: str, design_id: str) -> dict[str, object]:
    path_a, status_a = _engine_structure_for_design(run_a_dir, engine_label, design_id)
    path_b, status_b = _engine_structure_for_design(run_b_dir, engine_label, design_id)
    if path_a is None or path_b is None:
        return {
            "binder_id": design_id,
            "prediction_geometry_status": f"missing prediction: run1={status_a}; run2={status_b}",
        }
    reference_path, reference_chains, reference_label = _target_reference_for_design(run_a_dir, design_id)
    if reference_path is None or not reference_chains:
        return {"binder_id": design_id, "prediction_geometry_status": "missing run 1 target reference"}
    try:
        reference_structure = _parse_structure_file(reference_path)
        target_chains_a = _run_target_chains(run_a_dir)
        target_chains_b = _run_target_chains(run_b_dir)
        aligned_text_a, target_rmsd_a, note_a = _aligned_structure_text(
            path_a,
            reference_structure,
            reference_chains,
            target_chains_a,
        )
        aligned_text_b, target_rmsd_b, note_b = _aligned_structure_text(
            path_b,
            reference_structure,
            reference_chains,
            target_chains_b,
        )
        if not aligned_text_a or not aligned_text_b:
            return {
                "binder_id": design_id,
                "prediction_geometry_status": f"alignment failed: run1={note_a}; run2={note_b}",
            }
        aligned_a = _pdb_text_to_structure(aligned_text_a, f"{_safe_sort_token(engine_label)}_{_safe_sort_token(design_id)}_a")
        aligned_b = _pdb_text_to_structure(aligned_text_b, f"{_safe_sort_token(engine_label)}_{_safe_sort_token(design_id)}_b")
        binder_atoms_a = _target_ca_atoms(aligned_a, ["A"])
        binder_atoms_b = _target_ca_atoms(aligned_b, ["A"])
        binder_count = min(len(binder_atoms_a), len(binder_atoms_b))
        binder_rmsd = _atom_rmsd(list(zip(binder_atoms_a[:binder_count], binder_atoms_b[:binder_count])))
        return {
            "binder_id": design_id,
            "prediction_geometry_status": "ok",
            "prediction_reference": f"{reference_label}: {reference_path.name} chains {','.join(reference_chains)}",
            "predicted_target_ca_rmsd_run1": target_rmsd_a,
            "predicted_target_ca_rmsd_run2": target_rmsd_b,
            "predicted_binder_ca_rmsd": binder_rmsd,
            "predicted_binder_ca_count": binder_count,
            "prediction_path_run1": str(path_a),
            "prediction_path_run2": str(path_b),
        }
    except Exception as exc:
        return {"binder_id": design_id, "prediction_geometry_status": f"failed: {exc}"}


def _prediction_overlay_for_design(
    run_a: dict[str, object],
    run_b: dict[str, object],
    engine_label: str,
    design_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], float | None]:
    run_rows = [run_a, run_b]
    run_a_dir = Path(str(run_a.get("run_dir") or ""))
    reference_path, reference_chains, reference_label = _target_reference_for_design(run_a_dir, design_id)
    status_rows: list[dict[str, object]] = []
    if reference_path is None or not reference_chains:
        return [], [
            {
                "engine": engine_label,
                "design": design_id,
                "status": "missing run 1 target reference",
            }
        ], None
    try:
        reference_structure = _parse_structure_file(reference_path)
    except Exception as exc:
        return [], [
            {
                "engine": engine_label,
                "design": design_id,
                "status": f"could not parse target reference: {exc}",
                "reference": str(reference_path),
            }
        ], None

    entries: list[dict[str, object]] = []
    aligned_structures = []
    colors = ["#0072B2", "#D55E00"]
    target_colors = ["#8a8f98", "#d1d5db"]
    for run_index, run_row in enumerate(run_rows):
        run_dir = Path(str(run_row.get("run_dir") or ""))
        path, status = _engine_structure_for_design(run_dir, engine_label, design_id)
        target_chains = _run_target_chains(run_dir)
        status_row = {
            "engine": engine_label,
            "run": f"{run_row.get('job_code')} | {run_row.get('target_id')}",
            "design": design_id,
            "target_chains": ",".join(target_chains),
            "reference": f"{reference_label}: {reference_path.name} chains {','.join(reference_chains)}",
            "status": status,
            "path": str(path or ""),
        }
        if path is not None:
            aligned_text, target_rmsd, note = _aligned_structure_text(
                path,
                reference_structure,
                reference_chains,
                target_chains,
            )
            status_row["alignment"] = note
            status_row["target_ca_rmsd"] = target_rmsd
            if aligned_text:
                entries.append(
                    {
                        "pdb": aligned_text,
                        "targetChains": reference_chains,
                        "label": f"run {run_index + 1} binder",
                        "binder_color": colors[run_index],
                        "target_color": target_colors[run_index],
                    }
                )
                try:
                    aligned_structures.append(
                        _pdb_text_to_structure(
                            aligned_text,
                            f"{_safe_sort_token(engine_label)}_{_safe_sort_token(design_id)}_run_{run_index + 1}",
                        )
                    )
                except Exception:
                    pass
        status_rows.append(status_row)

    binder_rmsd = None
    if len(aligned_structures) == 2:
        binder_rmsd = _atom_rmsd(
            list(
                zip(
                    _target_ca_atoms(aligned_structures[0], ["A"]),
                    _target_ca_atoms(aligned_structures[1], ["A"]),
                )
            )
        )
        for row in status_rows:
            row["predicted_binder_ca_rmsd"] = binder_rmsd
    return entries, status_rows, binder_rmsd


def _selected_binder_from_altair_event(event: object) -> str | None:
    if event is None:
        return None
    if hasattr(event, "selection"):
        selection = getattr(event, "selection")
    elif isinstance(event, dict):
        selection = event.get("selection")
    else:
        selection = None
    if not isinstance(selection, dict):
        return None
    for value in selection.values():
        records = value if isinstance(value, list) else [value]
        for record in records:
            if isinstance(record, dict):
                binder_id = record.get("binder_id")
                if binder_id:
                    return str(binder_id)
    return None


@st.cache_data(show_spinner=False)
def _prediction_geometry_table(
    run_a_dir_text: str,
    run_b_dir_text: str,
    engine_label: str,
    design_ids: tuple[str, ...],
) -> pd.DataFrame:
    run_a_dir = Path(run_a_dir_text)
    run_b_dir = Path(run_b_dir_text)
    return pd.DataFrame(
        [
            _prediction_geometry_for_design(run_a_dir, run_b_dir, engine_label, str(design_id))
            for design_id in design_ids
        ]
    )


def _run_target_chains(run_dir: Path) -> list[str]:
    row = _first_run_csv_row(run_dir)
    chains = _parse_chain_list(row.get("target_chains"))
    return chains or ["B"]


def _run_csv_row_for_design(run_dir: Path, design_id: str) -> dict[str, object]:
    run_csv = run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv"
    if not run_csv.exists():
        return {}
    try:
        table = pd.read_csv(run_csv)
    except Exception:
        return {}
    if table.empty:
        return {}
    if "binder_id" in table.columns:
        subset = table[table["binder_id"].astype(str).str.lower().eq(str(design_id).lower())]
        if not subset.empty:
            return subset.iloc[0].to_dict()
    return table.iloc[0].to_dict()


def _target_reference_for_design(run_dir: Path, design_id: str) -> tuple[Path | None, list[str], str]:
    row = _run_csv_row_for_design(run_dir, design_id)
    for path_column, chain_column, label in [
        ("target_pdb", "target_chains", "staged prediction target"),
        ("source_target_pdb", "source_target_chains", "source target"),
    ]:
        path = _resolve_run_relative_path(run_dir, row.get(path_column))
        chains = _parse_chain_list(row.get(chain_column))
        if path is not None and chains:
            return path, chains, label
    return None, [], "missing target reference"


def _py3dmol_overlay_html(
    entries: list[dict[str, object]],
    *,
    div_id: str,
    height: int,
    show_targets: bool,
    show_binders: bool,
) -> str:
    js_url = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
    entries_json = json.dumps(entries)
    return f"""
<div id="{div_id}" style="width:100%; height:{height}px; position:relative; border:1px solid #d8dee9; border-radius:6px; overflow:hidden;"></div>
<div style="display:flex; flex-wrap:wrap; gap:10px; margin:8px 0 0 0; font:12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
{''.join(
    f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{str(entry.get("binder_color") or "#888")};margin-right:5px;"></span>{str(entry.get("label") or "")}</div>'
    for entry in entries
)}
</div>
<script>
(function() {{
  const entries = {entries_json};
  const jsUrl = {json.dumps(js_url)};
  const showTargets = {str(bool(show_targets)).lower()};
  const showBinders = {str(bool(show_binders)).lower()};
  function loadScriptAsync(uri) {{
    return new Promise((resolve, reject) => {{
      if (window.$3Dmol) {{ resolve(); return; }}
      const tag = document.createElement('script');
      tag.src = uri;
      tag.async = true;
      tag.onload = resolve;
      tag.onerror = reject;
      document.head.appendChild(tag);
    }});
  }}
  loadScriptAsync(jsUrl).then(function() {{
    const container = document.getElementById({json.dumps(div_id)});
    if (!container) return;
    container.innerHTML = '';
    const viewer = $3Dmol.createViewer(container, {{backgroundColor: 'white'}});
    for (const entry of entries) {{
      const model = viewer.addModel(entry.pdb, 'pdb');
      if (showTargets && entry.targetChains && entry.targetChains.length) {{
        for (const chain of entry.targetChains) {{
          model.setStyle({{chain: chain}}, {{cartoon: {{color: entry.target_color, opacity: 0.45}}}});
        }}
      }}
      if (showBinders) {{
        if (entry.targetChains && entry.targetChains.length) {{
          model.setStyle({{not: {{chain: entry.targetChains}}}}, {{cartoon: {{color: entry.binder_color, opacity: 0.92}}}});
        }} else {{
          model.setStyle({{}}, {{cartoon: {{color: entry.binder_color, opacity: 0.92}}}});
        }}
      }}
    }}
    viewer.zoomTo();
    viewer.render();
  }}).catch(function() {{
    const container = document.getElementById({json.dumps(div_id)});
    if (container) {{
      container.innerHTML = '<p style="padding:12px; background:#fff3cd; color:#664d03;">3Dmol.js could not be loaded. Check browser network access to jsDelivr.</p>';
    }}
  }});
}})();
</script>
"""


def _py3dmol_overlay_grid_html(
    panels: list[dict[str, object]],
    *,
    div_id: str,
    height: int,
    show_targets: bool,
    show_binders: bool,
) -> str:
    panel_count = max(1, len(panels))
    cols = 2 if panel_count <= 4 else 3
    rows = max(1, (panel_count + cols - 1) // cols)
    js_url = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
    panels_json = json.dumps(panels)
    return f"""
<div id="{div_id}" style="width:100%; height:{height}px; position:relative; border:1px solid #d8dee9; border-radius:6px; overflow:hidden;"></div>
<div style="display:flex; flex-wrap:wrap; gap:10px; margin:8px 0 0 0; font:12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#0072B2;margin-right:5px;"></span>run 1 binder</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#D55E00;margin-right:5px;"></span>run 2 binder</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#8a8f98;margin-right:5px;"></span>targets</div>
</div>
<script>
(function() {{
  const panels = {panels_json};
  const jsUrl = {json.dumps(js_url)};
  const rows = {rows};
  const cols = {cols};
  const showTargets = {str(bool(show_targets)).lower()};
  const showBinders = {str(bool(show_binders)).lower()};
  function loadScriptAsync(uri) {{
    return new Promise((resolve, reject) => {{
      if (window.$3Dmol) {{ resolve(); return; }}
      const tag = document.createElement('script');
      tag.src = uri;
      tag.async = true;
      tag.onload = resolve;
      tag.onerror = reject;
      document.head.appendChild(tag);
    }});
  }}
  function styleEntry(model, entry, targetColor, binderColor) {{
    if (showTargets && entry.targetChains && entry.targetChains.length) {{
      for (const chain of entry.targetChains) {{
        model.setStyle({{chain: chain}}, {{cartoon: {{color: targetColor, opacity: 0.38}}}});
      }}
    }}
    if (showBinders) {{
      if (entry.targetChains && entry.targetChains.length) {{
        model.setStyle({{not: {{chain: entry.targetChains}}}}, {{cartoon: {{color: binderColor, opacity: 0.92}}}});
      }} else {{
        model.setStyle({{}}, {{cartoon: {{color: binderColor, opacity: 0.92}}}});
      }}
    }}
  }}
  loadScriptAsync(jsUrl).then(function() {{
    const container = document.getElementById({json.dumps(div_id)});
    if (!container) return;
    container.innerHTML = '';
    const grid = $3Dmol.createViewerGrid(container, {{rows: rows, cols: cols, control_all: true}}, {{backgroundColor: 'white'}});
    for (let idx = 0; idx < panels.length; idx += 1) {{
      const row = Math.floor(idx / cols);
      const col = idx % cols;
      const viewer = grid[row][col];
      const panel = panels[idx];
      const modelA = viewer.addModel(panel.runA.pdb, 'pdb');
      styleEntry(modelA, panel.runA, '#8a8f98', '#0072B2');
      const modelB = viewer.addModel(panel.runB.pdb, 'pdb');
      styleEntry(modelB, panel.runB, '#d1d5db', '#D55E00');
      viewer.zoomTo();
      viewer.render();
    }}
  }}).catch(function() {{
    const container = document.getElementById({json.dumps(div_id)});
    if (container) {{
      container.innerHTML = '<p style="padding:12px; background:#fff3cd; color:#664d03;">3Dmol.js could not be loaded. Check browser network access to jsDelivr.</p>';
    }}
  }});
}})();
</script>
<div style="display:grid; grid-template-columns:repeat({cols}, 1fr); gap:6px; margin-top:6px; font:12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:#374151;">
{''.join(f'<div style="text-align:center; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{str(panel.get("engine") or "")}</div>' for panel in panels)}
</div>
"""


dataset_tab, engines_tab, metrics_tab, run_tab, results_tab, compare_tab, legacy_tab = st.tabs(
    ["Dataset", "Engines", "Metrics", "Run", "Results", "Compare", "Legacy"]
)

with dataset_tab:
    st.subheader("Dataset")
    selected = st.selectbox("Candidate set", range(len(usable_sources)), format_func=lambda index: labels[index])
    source = usable_sources[selected]
    candidates = load_source_candidates(source)
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Candidates", int(source["candidate_count"]))
    with summary_cols[1]:
        st.metric("Stages", len(source["stage_counts"]))
    with summary_cols[2]:
        st.metric("Job", source["job_code"])
    with summary_cols[3]:
        st.metric("Tool", source["tool"])
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    selection_key = f"refolding_candidate_selection_{source['run_id']}"
    select_all_key = f"refolding_select_all_{source['run_id']}"
    editor_version_key = f"refolding_candidate_editor_version_{source['run_id']}"
    source_signature = str(source["run_id"])
    previous_signature_key = "refolding_candidate_source_signature"
    select_all_candidates = st.checkbox(
        "Select all candidates from this set",
        value=st.session_state.get(select_all_key, False),
        key=select_all_key,
        help="Checking selects every candidate in this source set. Unchecking clears the selection.",
    )
    available_ids = set(candidate_ids)
    previous_selection = set(st.session_state.get(selection_key, [])) & available_ids
    previous_select_all = bool(st.session_state.get(f"{select_all_key}_previous", False))
    previous_signature = st.session_state.get(previous_signature_key)
    selection_changed = select_all_candidates != previous_select_all or source_signature != previous_signature
    if selection_changed:
        if select_all_candidates:
            previous_selection = available_ids
        elif select_all_candidates != previous_select_all or source_signature != previous_signature:
            previous_selection = set()
        st.session_state[editor_version_key] = int(st.session_state.get(editor_version_key, 0)) + 1
    st.session_state[f"{select_all_key}_previous"] = select_all_candidates
    st.session_state[previous_signature_key] = source_signature
    candidate_rows = []
    for row in candidates:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        candidate_id = str(row.get("candidate_id") or "")
        candidate_rows.append(
            {
                "Select": bool(select_all_candidates or candidate_id in previous_selection),
                "candidate_id": candidate_id,
                "stage": row.get("stage"),
                "source_tool": row.get("source_tool"),
                "import_rank": metrics.get("import_rank"),
                "bindcraft_final_rank": metrics.get("bindcraft_final_rank") or metrics.get("bindcraft_Rank"),
                "binder_chains": ",".join(row.get("binder_chains") or []),
                "target_chains": ",".join(row.get("target_chains") or []),
                "complex_pdb": row.get("complex_pdb"),
                "binder_sequence": row.get("binder_sequence"),
            }
        )
    candidate_df = pd.DataFrame(candidate_rows)
    for rank_column in ["import_rank", "bindcraft_final_rank"]:
        if rank_column in candidate_df.columns:
            candidate_df[rank_column] = pd.to_numeric(candidate_df[rank_column], errors="coerce")
    display_cols = [
        "candidate_id",
        "stage",
        "source_tool",
        "import_rank",
        "bindcraft_final_rank",
        "binder_chains",
        "target_chains",
        "complex_pdb",
        "binder_sequence",
    ]
    display_cols = [
        column
        for column in display_cols
        if column in candidate_df.columns and not candidate_df[column].isna().all()
    ]
    sort_options = display_cols or ["candidate_id"]
    default_sort = next(
        (column for column in ["import_rank", "bindcraft_final_rank"] if column in sort_options),
        "candidate_id",
    )
    sort_cols = st.columns([2, 1, 4])
    sort_col = sort_cols[0].selectbox(
        "Candidate table order",
        sort_options,
        index=sort_options.index(default_sort),
        key=f"refolding_candidate_sort_col_{source['run_id']}",
        help="Use this instead of the grid header sort; checkbox edits rerun the page and reset browser-only sorting.",
    )
    sort_descending = sort_cols[1].checkbox(
        "Descending",
        value=False,
        key=f"refolding_candidate_sort_descending_{source['run_id']}",
    )
    ordered_candidates = candidate_df.copy()
    if sort_col in ordered_candidates.columns:
        sort_values = ordered_candidates[sort_col]
        if not pd.api.types.is_numeric_dtype(sort_values) and not pd.api.types.is_bool_dtype(sort_values):
            ordered_candidates = ordered_candidates.assign(_candidate_sort_value=sort_values.astype(str).str.lower())
            sort_by = "_candidate_sort_value"
        else:
            sort_by = sort_col
        ordered_candidates = ordered_candidates.sort_values(
            by=sort_by,
            ascending=not sort_descending,
            kind="mergesort",
            na_position="last",
        ).drop(columns=["_candidate_sort_value"], errors="ignore")
    editor_df = ordered_candidates[display_cols].copy()
    editor_df.insert(0, "Select", editor_df["candidate_id"].isin(previous_selection))
    previous_selected_ids = set(previous_selection)
    edited_candidates = st.data_editor(
        editor_df,
        width="stretch",
        height=520,
        hide_index=True,
        disabled=[column for column in editor_df.columns if column != "Select"],
        column_config={
            "Select": st.column_config.CheckboxColumn("Select"),
            "binder_sequence": st.column_config.TextColumn("binder_sequence", width="large"),
        },
        key=(
            f"refolding_candidate_table_{source['run_id']}_"
            f"{int(st.session_state.get(editor_version_key, 0))}_"
            f"{_safe_sort_token(sort_col)}_{int(sort_descending)}"
        ),
    )
    selected_candidate_ids = [
        str(row.get("candidate_id") or "")
        for row in edited_candidates.to_dict("records")
        if bool(row.get("Select")) and str(row.get("candidate_id") or "")
    ]
    st.session_state[selection_key] = selected_candidate_ids
    if set(selected_candidate_ids) != previous_selected_ids:
        st.session_state[editor_version_key] = int(st.session_state.get(editor_version_key, 0)) + 1
        st.rerun()
    st.info(f"{len(selected_candidate_ids):,} of {len(candidates):,} candidates selected for refolding.")

    with st.expander("Target Structure Override", expanded=False):
        st.caption(
            "Optional. Use this when the imported design complex contains a target conformation or construct "
            "that should not be used for the new refolding run. Binder sequences still come from the candidate table; "
            "the selected target PDB is staged as target chain A and onward, while binder chains are staged from Z backward."
        )
        target_library = _source_target_library(source, candidates) + _benchmark_target_library()
        if target_library:
            library_rows = []
            for entry in target_library:
                library_rows.append(
                    {
                        "target_id": entry.get("target_id"),
                        "source": entry.get("source"),
                        "origin": entry.get("origin"),
                        "records": entry.get("records"),
                        "target_chains": ",".join(entry.get("chains") or []),
                        "path": entry.get("path"),
                    }
                )
            st.dataframe(pd.DataFrame(library_rows), width="stretch", height=220, hide_index=True)
            selected_library_index = st.selectbox(
                "Target structure library",
                range(len(target_library)),
                format_func=lambda index: str(target_library[index].get("label") or target_library[index].get("path")),
                key=f"refolding_target_library_choice_{source['run_id']}",
                help="Targets from the selected candidate source are listed first; installed benchmark-set targets are listed after them.",
            )
            selected_library_entry = target_library[int(selected_library_index)]
        else:
            selected_library_entry = None
            st.info("No readable target structures were found in this candidate set or the installed benchmark target library.")
        override_enabled = st.checkbox(
            "Use a different target PDB for prediction inputs",
            value=False,
            key=f"refolding_target_override_enabled_{source['run_id']}",
        )
        override_path_key = f"refolding_target_override_pdb_{source['run_id']}"
        override_chains_key = f"refolding_target_override_chains_{source['run_id']}"
        override_library_signature_key = f"refolding_target_override_library_signature_{source['run_id']}"
        default_target_path = ""
        first_candidate = candidates[0] if candidates else {}
        if isinstance(first_candidate, dict) and first_candidate.get("target_pdb"):
            default_target_path = str(first_candidate.get("target_pdb") or "")
        if override_path_key not in st.session_state:
            st.session_state[override_path_key] = default_target_path
        if not override_enabled:
            if str(st.session_state.get(override_path_key) or "") != default_target_path:
                st.session_state[override_path_key] = default_target_path
            st.session_state[override_chains_key] = [
                str(chain)
                for chain in (first_candidate.get("target_chains") if isinstance(first_candidate, dict) else []) or []
                if str(chain)
            ]
            st.session_state.pop(override_library_signature_key, None)
        if selected_library_entry is not None:
            selected_library_signature = (
                f"{selected_library_entry.get('path') or ''}|"
                f"{','.join(str(chain) for chain in (selected_library_entry.get('chains') or []))}"
            )
            current_library_signature = st.session_state.get(override_library_signature_key)
            current_override_path = str(st.session_state.get(override_path_key) or "")
            if (
                override_enabled
                and selected_library_signature != current_library_signature
                and (not current_override_path or current_override_path == default_target_path or current_library_signature)
            ):
                st.session_state[override_path_key] = str(selected_library_entry.get("path") or "")
                st.session_state[override_chains_key] = list(selected_library_entry.get("chains") or [])
                st.session_state[override_library_signature_key] = selected_library_signature
                st.rerun()
            use_cols = st.columns([1, 5])
            if use_cols[0].button(
                "Use selected target",
                disabled=not override_enabled,
                key=f"refolding_use_library_target_{source['run_id']}",
            ):
                st.session_state[override_path_key] = str(selected_library_entry.get("path") or "")
                st.session_state[override_chains_key] = list(selected_library_entry.get("chains") or [])
                st.session_state[override_library_signature_key] = selected_library_signature
                st.rerun()
            use_cols[1].caption(
                f"Selected library entry: {selected_library_entry.get('target_id')} "
                f"from {selected_library_entry.get('origin')}"
            )
        override_cols = st.columns([3, 2])
        override_target_text = override_cols[0].text_input(
            "Target PDB path",
            disabled=not override_enabled,
            key=override_path_key,
            help="Can be the same target in a different construct/conformation, for example a full-length or benchmark-set target structure.",
        )
        override_target_path = Path(str(override_target_text)).expanduser() if str(override_target_text or "").strip() else None
        if override_target_path is not None and not override_target_path.is_absolute():
            override_target_path = Path(str(source["run_dir"])) / override_target_path
        override_chain_options: list[str] = []
        if override_enabled and override_target_path is not None and override_target_path.exists():
            try:
                override_chain_options = refolding_workflow._structure_chains(override_target_path)
            except Exception:
                override_chain_options = []
        current_override_chains = [
            str(chain)
            for chain in st.session_state.get(override_chains_key, [])
            if str(chain) in set(override_chain_options)
        ]
        if override_enabled and override_chain_options and not current_override_chains:
            st.session_state[override_chains_key] = override_chain_options
        elif current_override_chains != st.session_state.get(override_chains_key, []):
            st.session_state[override_chains_key] = current_override_chains
        override_target_chains = override_cols[1].multiselect(
            "Target chain(s)",
            override_chain_options,
            disabled=not override_enabled or not override_chain_options,
            key=override_chains_key,
            help="These source chains are copied into the engine inputs as A, B, and following target chains; binder chains are assigned from Z backward.",
        )
        if override_enabled:
            if override_target_path is None or not override_target_path.exists():
                st.warning("Target override is enabled, but the target PDB path is not readable.")
            elif not override_target_chains:
                st.warning("Target override is enabled, but no target chains are selected.")
            else:
                st.info(
                    f"Override target will be used for prediction: {override_target_path} "
                    f"chains {', '.join(override_target_chains)}."
                )
with engines_tab:
    st.subheader("Prediction / Refolding Engines")
    st.caption("These generate structures or confidence outputs. Shared interface metrics are configured separately.")

    refolding_engine_keys = [
        "refolding_eval_run_af3",
        "refolding_eval_run_colab",
        "refolding_eval_run_af2",
        "refolding_eval_run_esmfold2",
        "refolding_eval_run_boltz2",
        "refolding_eval_run_rf3",
        "refolding_eval_run_openfold3",
        "refolding_eval_run_protenix",
        "refolding_eval_run_protenix_v1",
        "refolding_eval_run_protenix_v2",
        "refolding_eval_run_boltzgen",
    ]
    if "refolding_engine_defaults_all_selected_v1" not in st.session_state:
        for engine_key in refolding_engine_keys:
            st.session_state[engine_key] = True
        st.session_state["refolding_engine_defaults_all_selected_v1"] = True
    if "refolding_engine_coordinate_defaults_v2" not in st.session_state:
        st.session_state["refolding_eval_run_boltzgen"] = False
        st.session_state["refolding_engine_coordinate_defaults_v2"] = True
    for engine_key in refolding_engine_keys:
        st.session_state.setdefault(engine_key, True)

    template_msa_engine_keys = {
        "refolding_eval_run_af3",
        "refolding_eval_run_colab",
        "refolding_eval_run_esmfold2",
        "refolding_eval_run_boltz2",
        "refolding_eval_run_rf3",
        "refolding_eval_run_protenix_v1",
        "refolding_eval_run_protenix_v2",
    }
    template_only_engine_keys = set(template_msa_engine_keys)
    template_only_engine_keys.add("refolding_eval_run_af2")
    msa_engine_keys = {
        "refolding_eval_run_af3",
        "refolding_eval_run_colab",
        "refolding_eval_run_esmfold2",
        "refolding_eval_run_boltz2",
        "refolding_eval_run_rf3",
        "refolding_eval_run_openfold3",
        "refolding_eval_run_protenix",
        "refolding_eval_run_protenix_v1",
        "refolding_eval_run_protenix_v2",
    }
    bulk_cols = st.columns([1, 1, 1.5, 1.5, 1.35, 2.65])
    if bulk_cols[0].button("Select all engines", key="refolding_select_all_engines"):
        for engine_key in refolding_engine_keys:
            st.session_state[engine_key] = True
        st.rerun()
    if bulk_cols[1].button("Deselect all engines", key="refolding_deselect_all_engines"):
        for engine_key in refolding_engine_keys:
            st.session_state[engine_key] = False
        st.rerun()
    if bulk_cols[2].button("Template + MSA engines", key="refolding_select_template_msa_engines"):
        for engine_key in refolding_engine_keys:
            st.session_state[engine_key] = engine_key in template_msa_engine_keys
        st.session_state["refolding_eval_af3_templates"] = True
        st.session_state["refolding_eval_af3_msa"] = True
        st.session_state["refolding_eval_colab_templates"] = True
        st.session_state["refolding_eval_colab_msa"] = True
        st.session_state["refolding_eval_af2_legacy_initial_guess"] = True
        st.session_state["refolding_eval_esm_modes"] = ["initial_guess"]
        st.session_state["refolding_eval_esm_msa"] = True
        st.session_state["refolding_eval_af2_binder_template"] = True
        st.session_state["refolding_eval_af2_interface_template"] = True
        st.session_state["refolding_eval_boltz_template"] = True
        st.session_state["refolding_eval_boltz_msa"] = True
        st.session_state["refolding_eval_rf3_template"] = True
        st.session_state["refolding_eval_rf3_msa"] = True
        st.session_state["refolding_eval_protenix_v1_template"] = True
        st.session_state["refolding_eval_protenix_v1_msa"] = True
        st.session_state["refolding_eval_protenix_v2_template"] = True
        st.session_state["refolding_eval_protenix_v2_msa"] = True
        st.session_state["refolding_eval_openfold3_msa"] = False
        st.session_state["refolding_eval_protenix_msa"] = False
        st.session_state["refolding_eval_require_real_target_msa"] = True
        st.rerun()
    if bulk_cols[3].button("Template-only engines", key="refolding_select_template_only_engines"):
        for engine_key in refolding_engine_keys:
            st.session_state[engine_key] = engine_key in template_only_engine_keys
        st.session_state["refolding_eval_af3_templates"] = True
        st.session_state["refolding_eval_af3_msa"] = False
        st.session_state["refolding_eval_colab_templates"] = True
        st.session_state["refolding_eval_colab_msa"] = False
        st.session_state["refolding_eval_af2_legacy_initial_guess"] = True
        st.session_state["refolding_eval_esm_modes"] = ["initial_guess"]
        st.session_state["refolding_eval_esm_msa"] = False
        st.session_state["refolding_eval_af2_binder_template"] = True
        st.session_state["refolding_eval_af2_interface_template"] = True
        st.session_state["refolding_eval_boltz_template"] = True
        st.session_state["refolding_eval_boltz_msa"] = False
        st.session_state["refolding_eval_rf3_template"] = True
        st.session_state["refolding_eval_rf3_msa"] = False
        st.session_state["refolding_eval_protenix_v1_template"] = True
        st.session_state["refolding_eval_protenix_v1_msa"] = False
        st.session_state["refolding_eval_protenix_v2_template"] = True
        st.session_state["refolding_eval_protenix_v2_msa"] = False
        st.session_state["refolding_eval_openfold3_msa"] = False
        st.session_state["refolding_eval_protenix_msa"] = False
        st.session_state["refolding_eval_require_real_target_msa"] = False
        st.rerun()
    if bulk_cols[4].button("MSA-only engines", key="refolding_select_msa_only_engines"):
        for engine_key in refolding_engine_keys:
            st.session_state[engine_key] = engine_key in msa_engine_keys
        st.session_state["refolding_eval_af3_templates"] = False
        st.session_state["refolding_eval_af3_msa"] = True
        st.session_state["refolding_eval_colab_templates"] = False
        st.session_state["refolding_eval_colab_msa"] = True
        st.session_state["refolding_eval_boltz_template"] = False
        st.session_state["refolding_eval_boltz_msa"] = True
        st.session_state["refolding_eval_rf3_template"] = False
        st.session_state["refolding_eval_rf3_msa"] = True
        st.session_state["refolding_eval_protenix_v1_template"] = False
        st.session_state["refolding_eval_protenix_v1_msa"] = True
        st.session_state["refolding_eval_protenix_v2_template"] = False
        st.session_state["refolding_eval_protenix_v2_msa"] = True
        st.session_state["refolding_eval_esm_modes"] = ["sequence"]
        st.session_state["refolding_eval_esm_msa"] = True
        st.session_state["refolding_eval_openfold3_msa"] = True
        st.session_state["refolding_eval_protenix_msa"] = True
        st.session_state["refolding_eval_require_real_target_msa"] = True
        st.rerun()
    if bulk_cols[5].button("No MSA + no template", key="refolding_disable_msa_template"):
        st.session_state["refolding_eval_run_af2"] = False
        st.session_state["refolding_eval_af3_templates"] = False
        st.session_state["refolding_eval_af3_msa"] = False
        st.session_state["refolding_eval_colab_templates"] = False
        st.session_state["refolding_eval_colab_msa"] = False
        st.session_state["refolding_eval_esm_modes"] = ["sequence"]
        st.session_state["refolding_eval_esm_msa"] = False
        st.session_state["refolding_eval_af2_legacy_initial_guess"] = True
        st.session_state["refolding_eval_af2_binder_template"] = False
        st.session_state["refolding_eval_af2_interface_template"] = False
        st.session_state["refolding_eval_boltz_template"] = False
        st.session_state["refolding_eval_boltz_msa"] = False
        st.session_state["refolding_eval_rf3_template"] = False
        st.session_state["refolding_eval_rf3_msa"] = False
        st.session_state["refolding_eval_openfold3_msa"] = False
        st.session_state["refolding_eval_protenix_msa"] = False
        st.session_state["refolding_eval_protenix_v1_template"] = False
        st.session_state["refolding_eval_protenix_v1_msa"] = False
        st.session_state["refolding_eval_protenix_v2_template"] = False
        st.session_state["refolding_eval_protenix_v2_msa"] = False
        st.session_state["refolding_eval_require_real_target_msa"] = False
        st.rerun()

    engine_cols = st.columns(11)
    with engine_cols[0]:
        eval_run_af3 = st.checkbox("AlphaFast AF3", value=True, key="refolding_eval_run_af3")
    with engine_cols[1]:
        eval_run_colab = st.checkbox("ColabFold", value=True, key="refolding_eval_run_colab")
    with engine_cols[2]:
        eval_run_af2 = st.checkbox(
            "AF2 target-only initial guess",
            value=True,
            key="refolding_eval_run_af2",
            help="Runs AF2 with initial-guess conditioning. Deselect AF2 for no-template/no-MSA comparisons.",
        )
    if eval_run_af2:
        st.session_state["refolding_eval_af2_legacy_initial_guess"] = True
    with engine_cols[3]:
        eval_run_esmfold2 = st.checkbox("ESMFold2", value=True, key="refolding_eval_run_esmfold2")
    with engine_cols[4]:
        eval_run_boltz2 = st.checkbox("Boltz-2", value=True, key="refolding_eval_run_boltz2")
    with engine_cols[5]:
        eval_run_rf3 = st.checkbox("RF3", value=True, key="refolding_eval_run_rf3")
    with engine_cols[6]:
        eval_run_openfold3 = st.checkbox("OpenFold-3", value=True, key="refolding_eval_run_openfold3")
    with engine_cols[7]:
        eval_run_protenix = st.checkbox("Protenix", value=True, key="refolding_eval_run_protenix")
    with engine_cols[8]:
        eval_run_protenix_v1 = st.checkbox("Protenix v1", value=False, key="refolding_eval_run_protenix_v1")
    with engine_cols[9]:
        eval_run_protenix_v2 = st.checkbox("Protenix v2", value=False, key="refolding_eval_run_protenix_v2")
    with engine_cols[10]:
        eval_run_boltzgen = st.checkbox(
            "BoltzGen fold (coordinate-conditioned)",
            value=False,
            key="refolding_eval_run_boltzgen",
            help="Explicit coordinate-conditioned mode. Unlike the standard engines, this consumes staged complex geometry.",
        )

    st.segmented_control(
        "Prediction input contract",
        ["engine_specific"],
        selection_mode="single",
        default="engine_specific",
        disabled=True,
        key="refolding_eval_input_mode_display",
        format_func={"engine_specific": "Binder sequence + selected target PDB"}.get,
        help="The original BindCraft complex is retained as a post-hoc reference. Coordinate-conditioned exceptions are explicitly labelled.",
    )
    selected_count = len(st.session_state.get(f"refolding_candidate_selection_{source['run_id']}", candidate_ids))
    st.caption(f"Prediction engines will process {selected_count:,} selected candidates.")

    eval_boltz_target_msa = bool(st.session_state.get("refolding_eval_boltz_msa", True))
    eval_af3_msa = bool(st.session_state.get("refolding_eval_af3_msa", True))
    eval_esm_msa = bool(st.session_state.get("refolding_eval_esm_msa", False))
    eval_rf3_msa = bool(st.session_state.get("refolding_eval_rf3_msa", True))
    eval_openfold3_msa = bool(st.session_state.get("refolding_eval_openfold3_msa", True))
    eval_protenix_msa = bool(st.session_state.get("refolding_eval_protenix_msa", True))
    eval_protenix_v1_msa = bool(st.session_state.get("refolding_eval_protenix_v1_msa", True))
    eval_protenix_v2_msa = bool(st.session_state.get("refolding_eval_protenix_v2_msa", True))
    eval_colab_msa_state = bool(st.session_state.get("refolding_eval_colab_msa", True))
    with st.expander("MSA Reference Data", expanded=True):
        msa_consuming_engine_selected = bool(
            (eval_run_af3 and eval_af3_msa)
            or (eval_run_colab and eval_colab_msa_state)
            or (eval_run_boltz2 and eval_boltz_target_msa)
            or (eval_run_esmfold2 and eval_esm_msa)
            or (eval_run_rf3 and eval_rf3_msa)
            or (eval_run_openfold3 and eval_openfold3_msa)
            or (eval_run_protenix and eval_protenix_msa)
            or (eval_run_protenix_v1 and eval_protenix_v1_msa)
            or (eval_run_protenix_v2 and eval_protenix_v2_msa)
        )
        msa_cols = st.columns(4)
        eval_msa_source = msa_cols[0].segmented_control(
            "Shared target MSA source",
            ["msa_repository_then_alphafast_mmseqs_gpu", "msa_repository", "alphafast_mmseqs_gpu", "repo_run_csv"],
            selection_mode="single",
            default="msa_repository_then_alphafast_mmseqs_gpu",
            disabled=not msa_consuming_engine_selected,
            format_func={
                "msa_repository_then_alphafast_mmseqs_gpu": "Repository, then AlphaFast/MMseqs",
                "msa_repository": "Repository only",
                "alphafast_mmseqs_gpu": "AlphaFast/MMseqs GPU",
                "repo_run_csv": "Existing run.csv paths",
            }.get,
            help="Prepared once before engine input generation. Target-chain MSAs can be adapted by AF3, ColabFold, Boltz-style inputs, RF3, Protenix, and ESMFold2; binder chains stay no-MSA.",
            key="refolding_eval_msa_source",
        )
        eval_msa_repository = msa_cols[1].text_input(
            "MSA repository",
            value=str(MSA_REPOSITORY_DIR),
            help="Shared target-chain MSA cache. Used first when repository-backed MSA mode is selected.",
            key="refolding_eval_msa_repository",
        )
        eval_alphafast_db = msa_cols[2].text_input(
            "Alignment/MMseqs DB dir",
            value=str(ALPHAFAST_DB_DIR),
            help="AlphaFast database root containing the MMseqs alignment databases.",
            key="refolding_eval_alphafast_db",
        )
        eval_alphafast_batch_size = msa_cols[3].number_input(
            "AlphaFast/MMseqs batch size",
            min_value=0,
            max_value=10000,
            value=0,
            step=1,
            key="refolding_eval_alphafast_batch_size",
            disabled=not msa_consuming_engine_selected,
            help="0 means auto/default. This controls AlphaFast data-pipeline batching for target MSA/template preparation.",
        )
        eval_require_real_target_msa = st.checkbox(
            "Require real target MSAs before prediction",
            value=True,
            key="refolding_eval_require_real_target_msa",
            disabled=not msa_consuming_engine_selected,
            help=(
                "When enabled, every selected target chain must have an A3M with more than the query sequence. "
                "Repository hits that are query-only are regenerated with AlphaFast/MMseqs when that source is enabled; "
                "if a real MSA still cannot be produced, the job stops before GPU prediction."
            ),
        )

    default_models = []
    if eval_run_af3 or eval_run_colab:
        default_models.append("af3")
    if eval_run_boltz2:
        default_models.append("boltz")
    if eval_run_colab:
        default_models.append("colabfold")
    eval_models = default_models or ["af3", "boltz", "colabfold"]

    with st.expander("AlphaFast AF3 Settings", expanded=eval_run_af3):
        af3_cols = st.columns(4)
        eval_alphafast_weights = af3_cols[0].text_input("AF3 weights dir", value=str(ALPHAFAST_WEIGHTS_DIR), disabled=not eval_run_af3, key="refolding_eval_alphafast_weights")
        eval_af3_recycles = af3_cols[1].number_input("AF3 recycles", min_value=1, max_value=48, value=10, step=1, key="refolding_eval_af3_recycles", disabled=not eval_run_af3)
        eval_af3_templates = af3_cols[2].checkbox(
            "Use templates",
            value=True,
            key="refolding_eval_af3_templates",
            disabled=not eval_run_af3,
            help="Embeds the staged input target chains as AF3 templates. Binder chains and binder-interface geometry are not templated.",
        )
        eval_af3_msa = af3_cols[3].checkbox(
            "Use target MSAs",
            value=eval_af3_msa,
            key="refolding_eval_af3_msa",
            disabled=not eval_run_af3,
            help="When off, AlphaFast AF3 runs with query-only/no target MSA input.",
        )

    with st.expander("ColabFold Settings", expanded=eval_run_colab):
        colab_cols = st.columns(5)
        eval_colab_cache = colab_cols[0].text_input("ColabFold / AF2 model cache", value=str(COLABFOLD_CACHE_DIR), disabled=not eval_run_colab, key="refolding_eval_colab_cache")
        eval_colab_recycles = colab_cols[1].number_input("ColabFold recycles", min_value=1, max_value=48, value=3, step=1, key="refolding_eval_colab_recycles", disabled=not eval_run_colab)
        eval_colab_models = colab_cols[2].number_input("ColabFold models", min_value=1, max_value=5, value=3, step=1, key="refolding_eval_colab_models", disabled=not eval_run_colab)
        eval_colab_templates = colab_cols[3].checkbox("Use templates", value=True, key="refolding_eval_colab_templates", disabled=not eval_run_colab, help="Uses the staged input target as the template. Binder chains and binder-interface geometry are not templated.")
        eval_colab_msa = colab_cols[4].checkbox("Use target MSAs", value=True, key="refolding_eval_colab_msa", disabled=not eval_run_colab, help="When off, ColabFold does not request or inject real target MSAs.")
        eval_colab_max_template_hits = 4

    with st.expander("AF2 Target-Only Initial Guess Settings", expanded=eval_run_af2):
        af2_cols = st.columns(4)
        eval_af2_recycles = af2_cols[0].number_input("AF2 recycles", min_value=1, max_value=24, value=3, step=1, key="refolding_eval_af2_recycles", disabled=not eval_run_af2)
        eval_af2_multimer = af2_cols[1].checkbox("AF2 multimer", value=True, key="refolding_eval_af2_multimer", disabled=not eval_run_af2)
        eval_af2_legacy_initial_guess = af2_cols[2].checkbox(
            "Whole-complex initial guess",
            value=True,
            key="refolding_eval_af2_legacy_initial_guess",
            disabled=True,
            help="AF2 runs in this workflow always use initial-guess conditioning. Deselect AF2 to omit it from no-template/no-MSA runs.",
        )
        eval_af2_binder_template = af2_cols[3].checkbox(
            "Binder/interface template (legacy)",
            value=False,
            key="refolding_eval_af2_binder_template",
            disabled=not eval_run_af2 or not eval_af2_legacy_initial_guess,
        )
        eval_af2_interface_template = st.checkbox(
            "Preserve template interface geometry (legacy)",
            value=False,
            key="refolding_eval_af2_interface_template",
            disabled=not eval_run_af2 or not eval_af2_legacy_initial_guess or not eval_af2_binder_template,
        )
        if not eval_af2_legacy_initial_guess:
            st.caption("Default: the selected target PDB is the target-only initial guess. Binder coordinates and the original BindCraft interface are not supplied to prediction.")

    with st.expander("ESMFold2 Settings", expanded=eval_run_esmfold2):
        eval_esm_modes = st.multiselect(
            "Modes",
            ["sequence", "initial_guess"],
            default=["initial_guess"],
            disabled=not eval_run_esmfold2,
            format_func={"sequence": "Sequence only", "initial_guess": "Selected-target distogram"}.get,
            key="refolding_eval_esm_modes",
            help="The structural mode conditions only on the selected target PDB. It never consumes BindCraft binder coordinates or interface geometry.",
        )
        esmfold2_preset_selector(
            key="refolding_eval",
            steps_key="refolding_eval_esm_steps",
            loops_key="refolding_eval_esm_loops",
            disabled=not eval_run_esmfold2,
        )
        esm_cols_eval = st.columns(5)
        esm_cols_eval[0].text_input("ESMFold2 model dir", value=str(ESMFOLD2_MODEL_DIR), disabled=True, key="refolding_eval_esm_model_dir")
        eval_esm_msa = esm_cols_eval[1].checkbox(
            "Use ESMFold2 target MSAs",
            value=False,
            key="refolding_eval_esm_msa",
            disabled=not eval_run_esmfold2,
            help=(
                "Default off. Pass prepared per-target-chain A3M files into ESMFold2 ProteinInput.msa. "
                "This is per-chain target MSA conditioning, not a ColabFold-style paired multimer A3M; "
                "missing, query-only, or mismatched MSAs are skipped and noted in the ESMFold2 metrics."
            ),
        )
        eval_esm_steps = esm_cols_eval[2].number_input("ESMFold2 sampling steps", min_value=1, max_value=256, value=68, step=1, key="refolding_eval_esm_steps", disabled=not eval_run_esmfold2)
        eval_esm_loops = esm_cols_eval[3].number_input("ESMFold2 recycling loops", min_value=1, max_value=64, value=10, step=1, key="refolding_eval_esm_loops", disabled=not eval_run_esmfold2)
        eval_esm_seed = esm_cols_eval[4].number_input("ESMFold2 seed", min_value=0, max_value=999999, value=0, step=1, key="refolding_eval_esm_seed", disabled=not eval_run_esmfold2)

    with st.expander("Boltz-2 Settings", expanded=eval_run_boltz2):
        boltz_cols = st.columns(6)
        boltz_cols[0].text_input("Boltz-2 model cache", value=str(BOLTZ_MODELS_DIR), disabled=True, key="refolding_eval_boltz_model_cache")
        eval_boltz_target_template = boltz_cols[1].checkbox("Use templates", value=True, key="refolding_eval_boltz_template", disabled=not eval_run_boltz2, help="Uses the staged input target as the template.")
        eval_boltz_target_msa = boltz_cols[2].checkbox("Use target MSAs", value=True, key="refolding_eval_boltz_msa", disabled=not eval_run_boltz2, help="Default on. Binder chains stay no-MSA; each declared target chain uses its prepared MSA when available.")
        eval_boltz_recycles = boltz_cols[3].number_input("Recycling steps", min_value=1, max_value=48, value=10, step=1, key="refolding_eval_boltz_recycles", disabled=not eval_run_boltz2)
        eval_boltz_sampling = boltz_cols[4].number_input("Sampling steps", min_value=1, max_value=1000, value=200, step=1, key="refolding_eval_boltz_sampling", disabled=not eval_run_boltz2)
        eval_boltz_samples = boltz_cols[5].number_input("Diffusion samples", min_value=1, max_value=20, value=3, step=1, key="refolding_eval_boltz_samples", disabled=not eval_run_boltz2)
        eval_boltz_write_full_pae = st.checkbox("Write full PAE", value=True, key="refolding_eval_boltz_write_full_pae", disabled=not eval_run_boltz2)

    with st.expander("RF3 Settings", expanded=eval_run_rf3):
        rf3_cols = st.columns(7)
        eval_rf3_checkpoint = rf3_cols[0].text_input("RF3 checkpoint", value=str(refolding_workflow.RF3_CHECKPOINT), key="refolding_eval_rf3_checkpoint", disabled=not eval_run_rf3)
        eval_rf3_template = rf3_cols[1].checkbox("Use templates", value=True, key="refolding_eval_rf3_template", disabled=not eval_run_rf3, help="Uses the staged input target chains as RF3 template coordinates.")
        eval_rf3_msa = rf3_cols[2].checkbox("Use target MSAs", value=True, key="refolding_eval_rf3_msa", disabled=not eval_run_rf3, help="Default on. Each declared target chain receives its prepared A3M; binder chains remain MSA-free.")
        eval_rf3_recycles = rf3_cols[3].number_input("RF3 recycles", min_value=1, max_value=48, value=10, step=1, key="refolding_eval_rf3_recycles", disabled=not eval_run_rf3)
        eval_rf3_steps = rf3_cols[4].number_input("RF3 diffusion steps", min_value=1, max_value=1000, value=50, step=1, key="refolding_eval_rf3_steps", disabled=not eval_run_rf3)
        eval_rf3_samples = rf3_cols[5].number_input("RF3 samples", min_value=1, max_value=20, value=5, step=1, key="refolding_eval_rf3_samples", disabled=not eval_run_rf3)
        eval_rf3_seed = rf3_cols[6].number_input("RF3 seed", min_value=0, max_value=999999, value=0, step=1, key="refolding_eval_rf3_seed", disabled=not eval_run_rf3)
        st.caption("RF3 folds the complex from sequences plus optional target-chain templates and per-chain target MSAs.")

    with st.expander("OpenFold-3 Settings", expanded=eval_run_openfold3):
        openfold_cols = st.columns(6)
        eval_openfold3_checkpoint = openfold_cols[0].text_input(
            "OpenFold-3 checkpoint",
            value=str(refolding_workflow.OPENFOLD3_CHECKPOINT),
            key="refolding_eval_openfold3_checkpoint",
            disabled=not eval_run_openfold3,
            help="Default shared path: /mnt/db/reference_files/openfold3/of3-p2-155k.pt.",
        )
        eval_openfold3_msa = openfold_cols[1].checkbox(
            "Use target MSAs",
            value=True,
            key="refolding_eval_openfold3_msa",
            disabled=not eval_run_openfold3,
            help="Default on. Binder chains remain MSA-free; declared target chains use prepared A3M files when available.",
        )
        eval_openfold3_samples = openfold_cols[2].number_input(
            "Diffusion samples",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="refolding_eval_openfold3_samples",
            disabled=not eval_run_openfold3,
        )
        eval_openfold3_seeds = openfold_cols[3].number_input(
            "Model seeds",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            key="refolding_eval_openfold3_seeds",
            disabled=not eval_run_openfold3,
        )
        eval_openfold3_recycles = openfold_cols[4].number_input(
            "Recycles",
            min_value=1,
            max_value=48,
            value=3,
            step=1,
            key="refolding_eval_openfold3_recycles",
            disabled=not eval_run_openfold3,
            help="OpenFold-3 default architecture.shared.num_recycles is 3.",
        )
        eval_openfold3_msa_server = openfold_cols[5].checkbox(
            "Use MSA server",
            value=False,
            key="refolding_eval_openfold3_msa_server",
            disabled=not eval_run_openfold3,
            help="Default off for local/high-throughput runs. Prefer the app's shared target-MSA repository.",
        )
        st.caption("OpenFold-3 runs from chain sequences. Target MSAs are attached as precomputed main MSAs; binder chains are supplied without MSAs.")

    with st.expander("Protenix Settings", expanded=eval_run_protenix):
        protenix_cols = st.columns(4)
        eval_protenix_msa = protenix_cols[0].checkbox("Use target MSAs", value=True, key="refolding_eval_protenix_msa", disabled=not eval_run_protenix, help="Default on. Reuses the app's prepared target-chain A3Ms, packaged as Protenix MSA directories.")
        eval_protenix_cycle = protenix_cols[1].number_input("Pairformer cycles", min_value=1, max_value=48, value=3, step=1, key="refolding_eval_protenix_cycle", disabled=not eval_run_protenix)
        eval_protenix_steps = protenix_cols[2].number_input("Diffusion steps", min_value=1, max_value=1000, value=50, step=1, key="refolding_eval_protenix_steps", disabled=not eval_run_protenix)
        eval_protenix_samples = protenix_cols[3].number_input("Samples", min_value=1, max_value=20, value=5, step=1, key="refolding_eval_protenix_samples", disabled=not eval_run_protenix)
        st.caption("Legacy PXDesign-backed Protenix v0.5 adapter. New standalone Protenix models are exposed separately below.")

    with st.expander("Protenix v1 Settings", expanded=eval_run_protenix_v1):
        protenix_v1_cols = st.columns(6)
        eval_protenix_v1_model = protenix_v1_cols[0].selectbox(
            "Model",
            [
                refolding_workflow.PROTENIX_V1_MODEL,
                refolding_workflow.PROTENIX_V1_20250630_MODEL,
            ],
            index=0,
            disabled=not eval_run_protenix_v1,
            key="refolding_eval_protenix_v1_model",
        )
        eval_protenix_v1_msa = protenix_v1_cols[1].checkbox("Use target MSAs", value=True, key="refolding_eval_protenix_v1_msa", disabled=not eval_run_protenix_v1)
        eval_protenix_v1_template = protenix_v1_cols[2].checkbox("Use templates", value=False, key="refolding_eval_protenix_v1_template", disabled=not eval_run_protenix_v1)
        eval_protenix_v1_cycle = protenix_v1_cols[3].number_input("Pairformer cycles", min_value=1, max_value=48, value=10, step=1, key="refolding_eval_protenix_v1_cycle", disabled=not eval_run_protenix_v1)
        eval_protenix_v1_steps = protenix_v1_cols[4].number_input("Diffusion steps", min_value=1, max_value=1000, value=200, step=1, key="refolding_eval_protenix_v1_steps", disabled=not eval_run_protenix_v1)
        eval_protenix_v1_samples = protenix_v1_cols[5].number_input("Samples", min_value=1, max_value=20, value=5, step=1, key="refolding_eval_protenix_v1_samples", disabled=not eval_run_protenix_v1)
        st.caption("Standalone Protenix CLI. Checkpoints and cache are mounted at /mnt/db/reference_files/protenix.")

    with st.expander("Protenix v2 Settings", expanded=eval_run_protenix_v2):
        protenix_v2_cols = st.columns(6)
        eval_protenix_v2_model = protenix_v2_cols[0].text_input("Model", value=refolding_workflow.PROTENIX_V2_MODEL, disabled=not eval_run_protenix_v2, key="refolding_eval_protenix_v2_model")
        eval_protenix_v2_msa = protenix_v2_cols[1].checkbox("Use target MSAs", value=True, key="refolding_eval_protenix_v2_msa", disabled=not eval_run_protenix_v2)
        eval_protenix_v2_template = protenix_v2_cols[2].checkbox("Use templates", value=False, key="refolding_eval_protenix_v2_template", disabled=not eval_run_protenix_v2)
        eval_protenix_v2_cycle = protenix_v2_cols[3].number_input("Pairformer cycles", min_value=1, max_value=48, value=10, step=1, key="refolding_eval_protenix_v2_cycle", disabled=not eval_run_protenix_v2)
        eval_protenix_v2_steps = protenix_v2_cols[4].number_input("Diffusion steps", min_value=1, max_value=1000, value=200, step=1, key="refolding_eval_protenix_v2_steps", disabled=not eval_run_protenix_v2)
        eval_protenix_v2_samples = protenix_v2_cols[5].number_input("Samples", min_value=1, max_value=20, value=5, step=1, key="refolding_eval_protenix_v2_samples", disabled=not eval_run_protenix_v2)
        st.caption("Standalone Protenix v2 CLI adapter using the shared /mnt/db/reference_files/protenix cache.")

    with st.expander("BoltzGen Fold Settings", expanded=eval_run_boltzgen):
        boltzgen_cols = st.columns(3)
        eval_boltzgen_recycles = boltzgen_cols[0].number_input("Recycling steps", min_value=1, max_value=48, value=3, step=1, key="refolding_eval_boltzgen_recycles", disabled=not eval_run_boltzgen)
        eval_boltzgen_sampling = boltzgen_cols[1].number_input("Sampling steps", min_value=1, max_value=1000, value=200, step=1, key="refolding_eval_boltzgen_sampling", disabled=not eval_run_boltzgen)
        eval_boltzgen_samples = boltzgen_cols[2].number_input("Diffusion samples", min_value=1, max_value=20, value=5, step=1, key="refolding_eval_boltzgen_samples", disabled=not eval_run_boltzgen)
        st.caption("BoltzGen uses the declared target chains as a structural template and predicts the designed binder region. It does not use an MSA in this stage.")

with metrics_tab:
    st.subheader("Metrics")
    st.caption(
        "Refolding reports native engine confidence separately from recalculated interface scores and "
        "coordinate-based Rosetta/PyMOL evaluation."
    )
    metric_cols = st.columns(4)
    eval_common = metric_cols[0].checkbox(
        "Recalculated interface scores",
        value=True,
        key="refolding_eval_common_metrics",
        help=(
            "Runs the shared post-processing layer for compatible engine outputs: ipSAE, LIS, "
            "pDockQ/pDockQ2, and derived interface scores from available PAE/contact data."
        ),
    )
    metric_cols[1].checkbox("LIS", value=True, disabled=True, key="refolding_eval_lis_display")
    metric_cols[2].checkbox("pDockQ / pDockQ2", value=True, disabled=True, key="refolding_eval_pdockq_display")
    metric_cols[3].checkbox(
        "Native engine confidence",
        value=True,
        disabled=True,
        key="refolding_eval_native_confidence_display",
        help="Always parsed when the selected engine writes native confidence values such as pTM, ipTM, pLDDT, ranking scores, or engine-level PAE summaries.",
    )
    st.caption(
        "Native confidence stays in the Engine Confidence / PAE result group. The optional recalculated "
        "interface scores stay in the Recalculated PAE / Interface result group."
    )
    extra_cols = st.columns(4)
    eval_rosetta_input = extra_cols[0].checkbox("Rosetta input metrics", value=True, key="refolding_eval_rosetta_input_metrics", help="Runs PyRosetta interface metrics for the staged original/input complexes.")
    eval_rosetta = extra_cols[1].checkbox("Rosetta predicted metrics", value=True, key="refolding_eval_rosetta_metrics", help="Stages available predicted structures as PDBs, then relaxes and scores them with PyRosetta.")
    eval_rosetta_cores = extra_cols[2].number_input("PyRosetta processes", min_value=1, max_value=64, value=32, step=1, key="refolding_eval_rosetta_cores", disabled=not (eval_rosetta_input or eval_rosetta), help="Used for all enabled Rosetta feature calculations.")
    eval_pymol = extra_cols[3].checkbox("PyMOL metrics", value=True, key="refolding_eval_pymol_metrics", help="Calculates SASA, interface residue, hydrogen-bond, and secondary-structure metrics for input and predicted PDB folders.")

with run_tab:
    st.subheader("Run")
    eval_name = st.text_input("Evaluation name", value=f"{source['job_code']} refolding evaluation", key="refolding_eval_name")
    selected_engines = [
        label
        for label, enabled in [
            ("AF3", eval_run_af3),
            ("ColabFold", eval_run_colab),
            (
                "AF2 whole-complex IG (legacy)"
                if eval_af2_legacy_initial_guess
                else "AF2 target-only IG",
                eval_run_af2,
            ),
            ("Boltz-2", eval_run_boltz2),
            ("ESMFold2", eval_run_esmfold2),
            ("RF3", eval_run_rf3),
            ("OpenFold-3", eval_run_openfold3),
            ("Protenix", eval_run_protenix),
            ("Protenix v1", eval_run_protenix_v1),
            ("Protenix v2", eval_run_protenix_v2),
            ("BoltzGen Fold", eval_run_boltzgen),
        ]
        if enabled
    ]
    selected_candidate_ids = list(st.session_state.get(f"refolding_candidate_selection_{source['run_id']}", candidate_ids))
    run_count = len(selected_candidate_ids)
    st.info(f"{run_count:,} selected candidates from {source['job_code']} will run through: {', '.join(selected_engines) or 'no engines selected'}.")
    with st.expander("Compute", expanded=True):
        eval_gpu = gpu_run_panel(key="refolding_eval", default="0")

    estimate_engine_keys: list[str] = []
    if eval_run_af3:
        estimate_engine_keys.append("alphafast_af3")
    if eval_run_colab:
        estimate_engine_keys.append("colabfold")
    if eval_run_af2:
        estimate_engine_keys.append("af2_initial_guess")
    if eval_run_boltz2:
        estimate_engine_keys.append("boltz2_initial_guess")
    if eval_run_esmfold2:
        estimate_engine_keys.append("esmfold2")
    if eval_run_rf3:
        estimate_engine_keys.append("rf3")
    if eval_run_openfold3:
        estimate_engine_keys.append("openfold3")
    if eval_run_protenix:
        estimate_engine_keys.append("protenix")
    if eval_run_protenix_v1:
        estimate_engine_keys.append("protenix_v1")
    if eval_run_protenix_v2:
        estimate_engine_keys.append("protenix_v2")
    if eval_run_boltzgen:
        estimate_engine_keys.append("boltzgen_fold")
    if eval_common or eval_rosetta_input or eval_rosetta or eval_pymol:
        estimate_engine_keys.append("postprocessing")
    selected_candidate_set = set(selected_candidate_ids)
    selected_lengths = [
        _candidate_total_length(candidate)
        for candidate in candidates
        if str(candidate.get("candidate_id") or "") in selected_candidate_set
    ]
    selected_lengths = [length for length in selected_lengths if length is not None]
    estimate_total_residues = sum(selected_lengths) if selected_lengths else None
    requested_total_length = max(selected_lengths) if selected_lengths else None
    estimate_engine_params: dict[str, dict[str, int]] = {}
    if "esmfold2" in estimate_engine_keys:
        estimate_engine_params["esmfold2"] = {
            "num_loops": int(eval_esm_loops),
            "num_sampling_steps": int(eval_esm_steps),
        }
    if estimate_engine_keys and run_count:
        estimate = estimate_engines(
            engines=estimate_engine_keys,
            candidate_count=run_count,
            total_residues=int(estimate_total_residues) if estimate_total_residues else None,
            engine_params=estimate_engine_params,
        )
        with st.expander("Runtime estimate", expanded=True):
            runtime_cols = st.columns(4)
            runtime_cols[0].metric("Estimated total", estimate["total_time"])
            runtime_cols[1].metric("Candidates", f"{run_count:,}")
            runtime_cols[2].metric("Residues", f"{int(estimate_total_residues):,}" if estimate_total_residues else "n/a")
            runtime_cols[3].metric("Engines/steps", len(estimate_engine_keys))
            estimate_df = _estimate_rows_dataframe(estimate)
            if not estimate_df.empty:
                st.dataframe(
                    estimate_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "seconds_per_candidate": st.column_config.NumberColumn("sec / candidate", format="%.1f"),
                        "seconds_per_residue": st.column_config.NumberColumn("sec / residue", format="%.3f"),
                    },
                )
            st.caption(
                "Estimates use previous completed jobs when available and fallback rates otherwise. "
                "They improve as more refolding and benchmark runs finish."
            )
    elif estimate_engine_keys:
        st.caption("Runtime estimate needs at least one selected candidate. Select candidates in the Dataset tab.")
    capacity_engine_keys = [engine for engine in estimate_engine_keys if engine != "postprocessing"]
    capacity_preset = "full_workflow" if (eval_common or eval_rosetta_input or eval_rosetta or eval_pymol) else "practical"
    if capacity_engine_keys and requested_total_length:
        with st.expander("Capacity warning", expanded=True):
            st.caption(
                f"Largest selected system: {requested_total_length:,} residues. "
                "Warnings use completed Capacity Benchmark runs with the same GPU and comparable run depth."
            )
            _show_capacity_warnings(
                capacity_warnings(
                    engine_keys=capacity_engine_keys,
                    total_length=int(requested_total_length),
                    matrix_mode="sequence_copy_multimer",
                    preset=capacity_preset,
                    gpu_device=str(eval_gpu),
                )
            )
    run_disabled = run_count == 0 or not selected_engines
    target_override_enabled = bool(st.session_state.get(f"refolding_target_override_enabled_{source['run_id']}", False))
    target_override_text = str(st.session_state.get(f"refolding_target_override_pdb_{source['run_id']}", "") or "").strip()
    target_override_path = Path(target_override_text).expanduser() if target_override_text else None
    if target_override_path is not None and not target_override_path.is_absolute():
        target_override_path = Path(str(source["run_dir"])) / target_override_path
    target_override_chains = list(st.session_state.get(f"refolding_target_override_chains_{source['run_id']}", []) or [])
    if target_override_enabled:
        if target_override_path is None or not target_override_path.exists() or not target_override_chains:
            run_disabled = True
            st.warning("Target override is enabled. Provide a readable target PDB and at least one target chain before running.")
    if st.button("Run refolding evaluation", type="primary", key="run_refolding_engine_evaluation", disabled=run_disabled):
        try:
            run_dir = benchmark_workflow.enqueue_candidate_refolding_evaluation(
                source_run_dir=Path(str(source["run_dir"])),
                candidates_jsonl=Path(str(source["candidates_jsonl"])),
                selected_candidate_ids=selected_candidate_ids,
                max_candidates=0,
                target_override_pdb=target_override_path if target_override_enabled else None,
                target_override_chains=target_override_chains if target_override_enabled else [],
                evaluation_name=str(eval_name or "Refolding evaluation"),
                models=list(eval_models or []),
                run_common_interface_metrics=bool(eval_common),
                run_pyrosetta_input_metrics=bool(eval_rosetta_input),
                run_predicted_rosetta_metrics=bool(eval_rosetta),
                run_pymol_metrics=bool(eval_pymol),
                pyrosetta_nprocs=int(eval_rosetta_cores),
                run_alphafast_af3=bool(eval_run_af3),
                run_colabfold=bool(eval_run_colab),
                run_af2_initial_guess=bool(eval_run_af2),
                run_boltz2_initial_guess=bool(eval_run_boltz2),
                run_esmfold2=bool(eval_run_esmfold2),
                run_rf3=bool(eval_run_rf3),
                run_openfold3=bool(eval_run_openfold3),
                run_protenix=bool(eval_run_protenix),
                run_protenix_v1=bool(eval_run_protenix_v1),
                run_protenix_v2=bool(eval_run_protenix_v2),
                run_boltzgen_fold=bool(eval_run_boltzgen),
                colabfold_msa_source=str(eval_msa_source if eval_colab_msa else "repo_run_csv"),
                msa_repository_dir=Path(str(eval_msa_repository)),
                require_real_target_msa=bool(eval_require_real_target_msa),
                alphafast_db_dir=Path(str(eval_alphafast_db)),
                alphafast_weights_dir=Path(str(eval_alphafast_weights)),
                colabfold_cache_dir=Path(str(eval_colab_cache)),
                alphafast_batch_size=int(eval_alphafast_batch_size),
                alphafast_num_recycles=int(eval_af3_recycles),
                alphafast_use_target_templates=bool(eval_af3_templates),
                alphafast_query_only_msa=not bool(eval_af3_msa),
                alphafast_gpu_device=eval_gpu,
                af2_num_recycles=int(eval_af2_recycles),
                af2_multimer=bool(eval_af2_multimer),
                af2_use_initial_guess=bool(eval_run_af2),
                af2_use_binder_template=bool(eval_af2_legacy_initial_guess and eval_af2_binder_template),
                af2_use_interface_template=bool(
                    eval_af2_legacy_initial_guess
                    and eval_af2_binder_template
                    and eval_af2_interface_template
                ),
                colabfold_num_recycles=int(eval_colab_recycles),
                colabfold_num_models=int(eval_colab_models),
                colabfold_gpu_device=eval_gpu,
                gpu_device=eval_gpu,
                colabfold_use_target_templates=bool(eval_colab_templates),
                colabfold_use_target_msa=bool(eval_colab_msa),
                colabfold_max_template_hits=int(eval_colab_max_template_hits),
                boltz2_use_target_template=bool(eval_boltz_target_template),
                boltz2_use_target_msa=bool(eval_boltz_target_msa),
                boltz2_recycling_steps=int(eval_boltz_recycles),
                boltz2_sampling_steps=int(eval_boltz_sampling),
                boltz2_diffusion_samples=int(eval_boltz_samples),
                boltz2_write_full_pae=bool(eval_boltz_write_full_pae),
                esmfold2_modes=list(eval_esm_modes or ["initial_guess"]),
                esmfold2_use_target_msa=bool(eval_esm_msa),
                num_sampling_steps=int(eval_esm_steps),
                num_loops=int(eval_esm_loops),
                seed=int(eval_esm_seed),
                rf3_checkpoint_path=Path(str(eval_rf3_checkpoint)),
                rf3_use_target_msa=bool(eval_rf3_msa),
                rf3_use_target_template=bool(eval_rf3_template),
                rf3_recycles=int(eval_rf3_recycles),
                rf3_num_steps=int(eval_rf3_steps),
                rf3_diffusion_batch_size=int(eval_rf3_samples),
                rf3_seed=int(eval_rf3_seed),
                openfold3_checkpoint_path=Path(str(eval_openfold3_checkpoint)),
                openfold3_use_target_msa=bool(eval_openfold3_msa),
                openfold3_num_diffusion_samples=int(eval_openfold3_samples),
                openfold3_num_model_seeds=int(eval_openfold3_seeds),
                openfold3_num_recycles=int(eval_openfold3_recycles),
                openfold3_use_msa_server=bool(eval_openfold3_msa_server),
                protenix_use_msa=bool(eval_protenix_msa),
                protenix_cycle=int(eval_protenix_cycle),
                protenix_diffusion_steps=int(eval_protenix_steps),
                protenix_samples=int(eval_protenix_samples),
                protenix_v1_model_name=str(eval_protenix_v1_model),
                protenix_v1_use_msa=bool(eval_protenix_v1_msa),
                protenix_v1_use_template=bool(eval_protenix_v1_template),
                protenix_v1_cycle=int(eval_protenix_v1_cycle),
                protenix_v1_diffusion_steps=int(eval_protenix_v1_steps),
                protenix_v1_samples=int(eval_protenix_v1_samples),
                protenix_v2_model_name=str(eval_protenix_v2_model),
                protenix_v2_use_msa=bool(eval_protenix_v2_msa),
                protenix_v2_use_template=bool(eval_protenix_v2_template),
                protenix_v2_cycle=int(eval_protenix_v2_cycle),
                protenix_v2_diffusion_steps=int(eval_protenix_v2_steps),
                protenix_v2_samples=int(eval_protenix_v2_samples),
                boltzgen_recycling_steps=int(eval_boltzgen_recycles),
                boltzgen_sampling_steps=int(eval_boltzgen_sampling),
                boltzgen_diffusion_samples=int(eval_boltzgen_samples),
            )
            spawn_worker_for_run(run_dir)
            st.success("Refolding evaluation queued. It will keep running independently of Streamlit.")
            show_pipeline_links(run_dir, [run_dir])
        except Exception as exc:
            st.error(str(exc))

with results_tab:
    st.subheader("Results")
    refresh_results_button("refolding_validation_refresh_results")
    rows = _refolding_result_rows()
    if rows:
        df = pd.DataFrame(rows)
        display_cols = [
            "result",
            "job_code",
            "run_id",
            "description",
            "status",
            "source_job_code",
            "source_run_id",
            "source_task_group",
            "design_set",
            "source_job",
            "candidates",
            "target_id",
            "target_source",
            "target_chains",
            "source_target_chains",
            "target_pdb",
            "target_length",
            "esmfold2_preset",
            "esmfold2_loops",
            "esmfold2_steps",
            "records",
            "current_phase",
            "current_engine",
            "created_at",
        ]
        display_df = df[[col for col in display_cols if col in df.columns]].copy()
        table_key = "refolding_results"
        event = st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            key=f"{table_key}_jobs_table",
            on_select="rerun",
            selection_mode="multi-row",
            column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
        )
        delete_result = st.session_state.pop(f"{table_key}_delete_result", None)
        if delete_result:
            level, message = delete_result
            if level == "success":
                st.success(str(message))
            else:
                st.error(str(message))

        table_widget_key = f"{table_key}_jobs_table"
        selected_indices = [idx for idx in selected_dataframe_rows(event, table_widget_key) if 0 <= idx < len(display_df)]
        selected_rows = display_df.iloc[selected_indices].copy() if selected_indices else display_df.iloc[0:0].copy()
        active_selected = selected_rows[selected_rows["status"].isin(ACTIVE_STATUSES)]
        if not active_selected.empty:
            st.warning("Running, queued, or preparing refolding evaluations cannot be deleted.")
        selected_rows = selected_rows[~selected_rows["status"].isin(ACTIVE_STATUSES)]
        selected_refs = [
            ("benchmark", str(row["run_id"]))
            for row in selected_rows.to_dict(orient="records")
        ]
        selected_refs_key = f"{table_key}_selected_delete_refs"
        if selected_refs:
            st.session_state[selected_refs_key] = selected_refs
        cached_selected_refs = st.session_state.get(selected_refs_key) or []
        delete_clicked = st.button(
            "Delete selected refolding evaluations",
            type="primary",
            disabled=not cached_selected_refs,
            key=f"{table_key}_request_delete_jobs",
        )
        if delete_clicked and cached_selected_refs:
            show_delete_jobs_dialog(
                table_key=table_key,
                pending_refs=cached_selected_refs,
                selected_refs_key=selected_refs_key,
                label="refolding evaluation",
            )
        else:
            st.caption("Select finished refolding-evaluation rows in the table to enable deletion.")
    else:
        st.info("No refolding-evaluation jobs yet.")

with compare_tab:
    st.subheader("Compare Refolding Runs")
    st.caption(
        "Select completed refolding runs to compare the same candidate designs across targets or engine settings. "
        "Input rank uses the original BindCraft rank when present, otherwise the selected candidate order."
    )
    rows = _refolding_result_rows()
    if not rows:
        st.info("No refolding-evaluation jobs yet.")
    else:
        all_runs = pd.DataFrame(rows)
        completed_runs = all_runs[all_runs["status"].eq("completed")].copy()
        if completed_runs.empty:
            st.info("No completed refolding-evaluation runs are available for comparison.")
        else:
            compare_display_cols = [
                "result",
                "job_code",
                "design_set",
                "records",
                "target_id",
                "target_source",
                "target_chains",
                "source_target_chains",
                "target_pdb",
                "esmfold2_preset",
                "created_at",
                "run_id",
            ]
            group_summary = (
                completed_runs.groupby("design_set", dropna=False)
                .agg(
                    runs=("run_id", "count"),
                    source_job=("source_job", "first"),
                    candidates=("candidates", "first"),
                    targets=("target_id", lambda values: ", ".join(sorted({str(value) for value in values if str(value)}))),
                    newest=("created_at", "max"),
                )
                .reset_index()
                .sort_values(["runs", "newest"], ascending=[False, False])
            )
            with st.expander("Available design sets", expanded=False):
                st.dataframe(group_summary, width="stretch", hide_index=True)
            design_sets = sorted(str(value) for value in completed_runs["design_set"].dropna().unique())
            selected_design_set = st.selectbox(
                "Design set to compare",
                design_sets,
                key="refolding_compare_design_set",
                help="Only runs made from the same selected candidate/design set are comparable design-by-design.",
            )
            filtered_runs = completed_runs[completed_runs["design_set"].astype(str).eq(str(selected_design_set))].copy()
            st.caption(f"{len(filtered_runs):,} completed refolding runs are available for this design set.")
            compare_display = filtered_runs[[col for col in compare_display_cols if col in filtered_runs.columns]].copy()
            compare_event = st.dataframe(
                compare_display,
                width="stretch",
                hide_index=True,
                key="refolding_compare_runs_table",
                on_select="rerun",
                selection_mode="multi-row",
                column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
            )
            selected_indices = [
                idx
                for idx in selected_dataframe_rows(compare_event, "refolding_compare_runs_table")
                if 0 <= idx < len(compare_display)
            ]
            selected_runs = filtered_runs.iloc[selected_indices].copy() if selected_indices else filtered_runs.iloc[0:0].copy()
            if len(selected_runs) < 2:
                st.info("Select two or more completed refolding runs to build a comparison.")
            else:
                loaded_frames: list[pd.DataFrame] = []
                run_labels: dict[str, str] = {}
                input_rank_maps: dict[str, dict[str, int]] = {}
                skipped: list[str] = []
                for row in selected_runs.to_dict(orient="records"):
                    run_dir = Path(str(row.get("run_dir") or ""))
                    metrics_csv = _metrics_csv_path(run_dir)
                    if not metrics_csv.exists():
                        skipped.append(str(row.get("job_code") or row.get("run_id")))
                        continue
                    try:
                        frame = pd.read_csv(metrics_csv)
                    except Exception:
                        skipped.append(str(row.get("job_code") or row.get("run_id")))
                        continue
                    if "binder_id" not in frame.columns:
                        skipped.append(str(row.get("job_code") or row.get("run_id")))
                        continue
                    run_key = str(row.get("run_id"))
                    run_label = f"{row.get('job_code')} | {row.get('target_id')} ({row.get('target_source')})"
                    run_labels[run_key] = run_label
                    input_rank_maps[run_key] = _candidate_input_rank_map(run_dir)
                    frame = frame.copy()
                    frame["_run_id"] = run_key
                    frame["_run_label"] = run_label
                    frame["_target_id"] = row.get("target_id")
                    frame["_target_source"] = row.get("target_source")
                    loaded_frames.append(frame)
                if skipped:
                    st.warning(f"Skipped runs without readable merged metrics: {', '.join(skipped)}.")
                if len(loaded_frames) < 2:
                    st.info("At least two selected runs need readable merged metrics.")
                else:
                    common_ids = set(str(value).lower() for value in loaded_frames[0]["binder_id"].dropna())
                    for frame in loaded_frames[1:]:
                        common_ids &= set(str(value).lower() for value in frame["binder_id"].dropna())
                    st.metric("Common designs", len(common_ids))
                    if not common_ids:
                        st.warning("The selected runs do not share binder IDs, so they cannot be compared directly.")
                    else:
                        filtered_common_ids = set(common_ids)
                        numeric_cols = _numeric_compare_columns(loaded_frames)
                        if not numeric_cols:
                            st.info("No shared numeric score columns were found across the selected runs.")
                        else:
                            presets = load_feature_presets()
                            preset_choices = ["Manual selectors"]
                            preset_by_label: dict[str, dict] = {}
                            for preset in presets:
                                preset_features = _preset_features_by_engine(preset, numeric_cols)
                                if not preset_features:
                                    continue
                                name = str(preset.get("name") or Path(str(preset.get("_path") or "preset")).stem)
                                label = f"Preset: {name}"
                                preset_choices.append(label)
                                preset_by_label[label] = preset
                            selected_preset_label = st.selectbox(
                                "Set engine score selectors from preset",
                                preset_choices,
                                key="refolding_compare_score_preset",
                            )
                            selected_preset = preset_by_label.get(selected_preset_label, {})
                            preset_features = _preset_features_by_engine(selected_preset, numeric_cols) if selected_preset else {}
                            preset_directions = _preset_directions_by_feature(selected_preset) if selected_preset else {}
                            preset_token = _safe_sort_token(selected_preset_label)
                            engine_to_features: dict[str, list[str]] = {}
                            for column in numeric_cols:
                                engine_to_features.setdefault(_feature_engine_name(column), []).append(column)
                            engines = sorted(engine_to_features, key=_engine_sort_key)
                            default_engines = [
                                engine
                                for engine in engines
                                if engine != "Other" and (not preset_features or engine in preset_features)
                            ]
                            selected_engines = st.multiselect(
                                "Engine panels",
                                engines,
                                default=default_engines or engines[: min(8, len(engines))],
                                key=f"refolding_compare_engine_panels_{preset_token}",
                            )
                            if preset_features:
                                preset_table = pd.DataFrame(
                                    [
                                        {"engine": engine, "preset_feature": feature}
                                        for engine, feature in sorted(preset_features.items(), key=lambda item: _engine_sort_key(item[0]))
                                    ]
                                )
                                st.dataframe(preset_table, width="stretch", hide_index=True)
                            all_comparison_rows = []
                            selected_run_labels = list(selected_runs.apply(
                                lambda row: f"{row.get('job_code')} | {row.get('target_id')} ({row.get('target_source')})",
                                axis=1,
                            ))
                            st.markdown("**Engine Ranking And Agreement Matrix**")
                            st.caption("Each engine panel keeps its ranking plot, run-agreement heatmap, and pairwise rank plot together.")
                            for engine in selected_engines:
                                features = sorted(engine_to_features.get(engine, []))
                                if not features:
                                    continue
                                default_feature = preset_features.get(engine)
                                if default_feature not in features:
                                    preferred = [
                                        "pDockQ2_min",
                                        "pDockQ_min",
                                        "ipae_min",
                                        "ipSAE_min_in_calculation",
                                        "ranking_score",
                                        "confidence_score",
                                        "interface_dG",
                                    ]
                                    default_feature = next(
                                        (
                                            feature
                                            for token in preferred
                                            for feature in features
                                            if token.lower() in feature.lower()
                                        ),
                                        features[0],
                                    )
                                with st.container(border=True):
                                    st.markdown(f"**{engine}**")
                                    feature = st.selectbox(
                                        f"{engine} ranking score",
                                        features,
                                        index=features.index(default_feature),
                                        key=f"refolding_compare_feature_{preset_token}_{_safe_sort_token(engine)}",
                                    )
                                    preset_direction = preset_directions.get(feature)
                                    default_ascending = preset_direction == "lower" if preset_direction else _rank_ascending(feature)
                                    ascending = st.checkbox(
                                        f"{engine}: lower score is better",
                                        value=default_ascending,
                                        key=f"refolding_compare_lower_{preset_token}_{_safe_sort_token(engine)}_{_safe_sort_token(feature)}",
                                    )
                                    engine_design_ids = set(filtered_common_ids)
                                    prediction_geometry_df = pd.DataFrame()
                                    first_rank_map = input_rank_maps.get(str(selected_runs.iloc[0].get("run_id"))) or {}
                                    engine_design_options = sorted(
                                        filtered_common_ids,
                                        key=lambda design: first_rank_map.get(str(design).lower(), 10**9),
                                    )
                                    selected_state_key = (
                                        f"refolding_compare_selected_design_state_{preset_token}_{_safe_sort_token(engine)}"
                                    )
                                    if st.session_state.get(selected_state_key) not in engine_design_options:
                                        st.session_state[selected_state_key] = engine_design_options[0]
                                    selected_index = engine_design_options.index(st.session_state[selected_state_key])
                                    selected_engine_design = st.selectbox(
                                        f"{engine} selected design",
                                        engine_design_options,
                                        index=selected_index,
                                        format_func=lambda design: f"#{first_rank_map.get(str(design).lower(), '?')} | {design}",
                                        key=(
                                            f"refolding_compare_selected_design_combo_{preset_token}_"
                                            f"{_safe_sort_token(engine)}_{_safe_sort_token(st.session_state[selected_state_key])}"
                                        ),
                                        help="This design is highlighted in this engine panel and used for the predicted complex overlay.",
                                    )
                                    if selected_engine_design != st.session_state[selected_state_key]:
                                        st.session_state[selected_state_key] = selected_engine_design
                                    if len(selected_runs) == 2 and engine in STRUCTURE_ENGINE_BY_LABEL:
                                        prediction_geometry_key = _safe_sort_token(
                                            f"{selected_runs.iloc[0].get('run_id')}_{selected_runs.iloc[1].get('run_id')}_{engine}_{preset_token}"
                                        )
                                        max_predicted_binder_rmsd = st.number_input(
                                            "Max predicted binder RMSD (A)",
                                            min_value=0.0,
                                            max_value=100.0,
                                            value=5.0,
                                            step=0.5,
                                            key=f"refolding_compare_pred_geom_max_{prediction_geometry_key}",
                                        )
                                        prediction_geometry_state_key = f"refolding_compare_pred_geometry_ready_{prediction_geometry_key}"
                                        if st.button(
                                            "Calculate predicted geometry",
                                            key=f"refolding_compare_pred_geometry_button_{prediction_geometry_key}",
                                            help="For this engine, aligns the two predicted complexes on the target and measures binder CA RMSD.",
                                        ):
                                            st.session_state[prediction_geometry_state_key] = True
                                        if st.session_state.get(prediction_geometry_state_key):
                                            with st.spinner(f"Calculating {engine} predicted geometry..."):
                                                prediction_geometry_df = _prediction_geometry_table(
                                                    str(selected_runs.iloc[0].get("run_dir") or ""),
                                                    str(selected_runs.iloc[1].get("run_dir") or ""),
                                                    engine,
                                                    tuple(sorted(filtered_common_ids)),
                                                )
                                            ok_prediction_geometry = prediction_geometry_df[
                                                prediction_geometry_df["prediction_geometry_status"].astype(str).eq("ok")
                                            ].copy()
                                            if not ok_prediction_geometry.empty:
                                                ok_prediction_geometry["predicted_binder_ca_rmsd"] = pd.to_numeric(
                                                    ok_prediction_geometry["predicted_binder_ca_rmsd"],
                                                    errors="coerce",
                                                )
                                                passing_prediction_geometry = ok_prediction_geometry[
                                                    ok_prediction_geometry["predicted_binder_ca_rmsd"].le(float(max_predicted_binder_rmsd))
                                                ].copy()
                                                pred_metric_cols = st.columns(2)
                                                pred_metric_cols[0].metric(
                                                    "Median predicted binder RMSD",
                                                    f"{ok_prediction_geometry['predicted_binder_ca_rmsd'].median():.2f} A",
                                                )
                                                pred_metric_cols[1].metric(
                                                    "Pass predicted RMSD",
                                                    f"{len(passing_prediction_geometry):,}",
                                                )
                                                use_prediction_filter = st.checkbox(
                                                    f"Filter {engine} plots by predicted binder RMSD",
                                                    value=False,
                                                    key=f"refolding_compare_pred_geometry_filter_{prediction_geometry_key}",
                                                )
                                                if use_prediction_filter:
                                                    engine_design_ids = set(passing_prediction_geometry["binder_id"].astype(str).str.lower())
                                                pred_geom_chart = (
                                                    alt.Chart(ok_prediction_geometry)
                                                    .mark_bar(opacity=0.82)
                                                    .encode(
                                                        x=alt.X("predicted_binder_ca_rmsd:Q", bin=alt.Bin(maxbins=24), title="predicted binder CA RMSD (A)"),
                                                        y=alt.Y("count():Q", title="designs"),
                                                        tooltip=[
                                                            alt.Tooltip("count():Q", title="designs"),
                                                        ],
                                                    )
                                                    .properties(height=120)
                                                )
                                                st.altair_chart(pred_geom_chart, width="stretch")
                                            else:
                                                st.warning(f"No {engine} predicted geometry diagnostics could be calculated.")
                                            with st.expander(f"{engine} predicted geometry table", expanded=False):
                                                st.dataframe(prediction_geometry_df, width="stretch", hide_index=True)
                                    comparison_rows = []
                                    for frame in loaded_frames:
                                        run_key = str(frame["_run_id"].iloc[0])
                                        subset = frame[frame["binder_id"].astype(str).str.lower().isin(engine_design_ids)].copy()
                                        subset[feature] = pd.to_numeric(subset[feature], errors="coerce")
                                        subset = subset.dropna(subset=[feature])
                                        subset["output_rank"] = subset[feature].rank(method="min", ascending=ascending).astype(int)
                                        rank_map = input_rank_maps.get(run_key) or {}
                                        for item in subset.to_dict(orient="records"):
                                            binder_id = str(item.get("binder_id") or "")
                                            input_rank = rank_map.get(binder_id.lower())
                                            comparison_rows.append(
                                                {
                                                    "engine": engine,
                                                    "feature": feature,
                                                    "run": item.get("_run_label"),
                                                    "target_id": item.get("_target_id"),
                                                    "target_source": item.get("_target_source"),
                                                    "binder_id": binder_id,
                                                    "input_rank": input_rank,
                                                    "output_rank": item.get("output_rank"),
                                                    "rank_delta": (item.get("output_rank") - input_rank) if input_rank else None,
                                                    "score": item.get(feature),
                                                }
                                            )
                                    engine_compare = pd.DataFrame(comparison_rows)
                                    if not prediction_geometry_df.empty and "binder_id" in prediction_geometry_df.columns:
                                        engine_compare = engine_compare.merge(
                                            prediction_geometry_df,
                                            on="binder_id",
                                            how="left",
                                        )
                                    if engine_compare.empty:
                                        st.info(f"No numeric {engine} values for `{feature}`.")
                                        continue
                                    all_comparison_rows.extend(comparison_rows)
                                    chart_data = engine_compare.dropna(subset=["output_rank", "score"]).copy()
                                    if not chart_data.empty:
                                        chart_data["selected_design"] = chart_data["binder_id"].astype(str).str.lower().eq(
                                            str(selected_engine_design).lower()
                                        )
                                        rank_base = (
                                            alt.Chart(chart_data)
                                            .mark_circle(size=42, opacity=0.78)
                                            .encode(
                                                x=alt.X("output_rank:Q", title=f"{feature} rank within each run"),
                                                y=alt.Y("score:Q", title=feature),
                                                color=alt.Color("run:N", title="run"),
                                                tooltip=[
                                                    "run:N",
                                                    "target_id:N",
                                                    "binder_id:N",
                                                    "input_rank:Q",
                                                    "output_rank:Q",
                                                    "rank_delta:Q",
                                                    alt.Tooltip("score:Q", title=feature),
                                                    alt.Tooltip("predicted_binder_ca_rmsd:Q", format=".2f"),
                                                ],
                                            )
                                        )
                                        rank_highlight = (
                                            alt.Chart(chart_data[chart_data["selected_design"]])
                                            .mark_circle(size=145, fillOpacity=0.15, stroke="#111827", strokeWidth=3)
                                            .encode(
                                                x=alt.X("output_rank:Q"),
                                                y=alt.Y("score:Q"),
                                                tooltip=[
                                                    "run:N",
                                                    "target_id:N",
                                                    "binder_id:N",
                                                    "input_rank:Q",
                                                    "output_rank:Q",
                                                    "rank_delta:Q",
                                                    alt.Tooltip("score:Q", title=feature),
                                                    alt.Tooltip("predicted_binder_ca_rmsd:Q", format=".2f"),
                                                ],
                                            )
                                        )
                                        chart = (rank_base + rank_highlight).properties(title="Engine rank vs selected score", height=260).interactive()
                                        st.altair_chart(chart, width="stretch")
                                    original_rank_data = engine_compare.dropna(subset=["input_rank", "output_rank"]).copy()
                                    if not original_rank_data.empty:
                                        original_rank_data["selected_design"] = original_rank_data["binder_id"].astype(str).str.lower().eq(
                                            str(selected_engine_design).lower()
                                        )
                                        original_rank_base = (
                                            alt.Chart(original_rank_data)
                                            .mark_circle(size=42, opacity=0.78)
                                            .encode(
                                                x=alt.X("input_rank:Q", title="original/input rank"),
                                                y=alt.Y("output_rank:Q", title=f"{feature} rank within each run"),
                                                color=alt.Color("run:N", title="run"),
                                                tooltip=[
                                                    "run:N",
                                                    "target_id:N",
                                                    "binder_id:N",
                                                    "input_rank:Q",
                                                    "output_rank:Q",
                                                    "rank_delta:Q",
                                                    alt.Tooltip("score:Q", title=feature),
                                                    alt.Tooltip("predicted_binder_ca_rmsd:Q", format=".2f"),
                                                ],
                                            )
                                        )
                                        original_rank_highlight = (
                                            alt.Chart(original_rank_data[original_rank_data["selected_design"]])
                                            .mark_circle(size=145, fillOpacity=0.15, stroke="#111827", strokeWidth=3)
                                            .encode(
                                                x=alt.X("input_rank:Q"),
                                                y=alt.Y("output_rank:Q"),
                                                tooltip=[
                                                    "run:N",
                                                    "target_id:N",
                                                    "binder_id:N",
                                                    "input_rank:Q",
                                                    "output_rank:Q",
                                                    "rank_delta:Q",
                                                    alt.Tooltip("score:Q", title=feature),
                                                    alt.Tooltip("predicted_binder_ca_rmsd:Q", format=".2f"),
                                                ],
                                            )
                                        )
                                        original_rank_chart = (
                                            (original_rank_base + original_rank_highlight)
                                            .properties(title="Original rank vs current engine rank", height=260)
                                            .interactive()
                                        )
                                        st.altair_chart(original_rank_chart, width="stretch")
                                    rank_matrix = engine_compare.pivot_table(
                                        index="binder_id",
                                        columns="run",
                                        values="output_rank",
                                        aggfunc="min",
                                    )
                                    if rank_matrix.shape[1] >= 2:
                                        correlation = rank_matrix.corr(method="spearman")
                                        correlation_rows = []
                                        for run_a in correlation.index:
                                            for run_b in correlation.columns:
                                                value = correlation.loc[run_a, run_b]
                                                if pd.isna(value):
                                                    continue
                                                correlation_rows.append(
                                                    {
                                                        "run_a": run_a,
                                                        "run_b": run_b,
                                                        "spearman": float(value),
                                                    }
                                                )
                                        if correlation_rows:
                                            corr_plot = pd.DataFrame(correlation_rows)
                                            heatmap = (
                                                alt.Chart(corr_plot)
                                                .mark_rect()
                                                .encode(
                                                    x=alt.X("run_a:N", title=None),
                                                    y=alt.Y("run_b:N", title=None),
                                                    color=alt.Color(
                                                        "spearman:Q",
                                                        title="Spearman",
                                                        scale=alt.Scale(domain=[-1, 0, 1], range=["#b2182b", "#f7f7f7", "#2166ac"]),
                                                    ),
                                                    tooltip=[
                                                        "run_a:N",
                                                        "run_b:N",
                                                        alt.Tooltip("spearman:Q", format=".3f"),
                                                    ],
                                                )
                                                .properties(title="Run agreement", height=220)
                                            )
                                            text = heatmap.mark_text(fontSize=11).encode(
                                                text=alt.Text("spearman:Q", format=".2f"),
                                                color=alt.condition(
                                                    "abs(datum.spearman) > 0.55",
                                                    alt.value("white"),
                                                    alt.value("#1f2937"),
                                                ),
                                            )
                                            st.altair_chart(heatmap + text, width="stretch")
                                    if len(selected_run_labels) == 2 and all(label in rank_matrix.columns for label in selected_run_labels):
                                        pair_df = rank_matrix[selected_run_labels].dropna().reset_index()
                                        pair_df.columns = ["binder_id", "run_a_rank", "run_b_rank"]
                                        pair_df["run_a_rank"] = pd.to_numeric(pair_df["run_a_rank"], errors="coerce")
                                        pair_df["run_b_rank"] = pd.to_numeric(pair_df["run_b_rank"], errors="coerce")
                                        pair_df = pair_df.dropna(subset=["run_a_rank", "run_b_rank"])
                                        if pair_df.empty:
                                            st.caption(f"{engine} pairwise rank agreement: no designs remain after the active filters.")
                                        else:
                                            pair_df["selected_design"] = pair_df["binder_id"].astype(str).str.lower().eq(
                                                str(selected_engine_design).lower()
                                            )
                                            max_pair_rank = float(max(pair_df["run_a_rank"].max(), pair_df["run_b_rank"].max()))
                                            point_selection = alt.selection_point(
                                                name="selected_design",
                                                fields=["binder_id"],
                                                on="click",
                                                empty=False,
                                            )
                                            pair_base = (
                                                alt.Chart(pair_df)
                                                .mark_circle(size=48, opacity=0.78)
                                                .encode(
                                                    x=alt.X(
                                                        "run_a_rank:Q",
                                                        title=selected_run_labels[0],
                                                        scale=alt.Scale(domain=[0, max_pair_rank + 1]),
                                                    ),
                                                    y=alt.Y(
                                                        "run_b_rank:Q",
                                                        title=selected_run_labels[1],
                                                        scale=alt.Scale(domain=[0, max_pair_rank + 1]),
                                                    ),
                                                    tooltip=["binder_id:N", "run_a_rank:Q", "run_b_rank:Q"],
                                                )
                                                .add_params(point_selection)
                                            )
                                            pair_highlight = (
                                                alt.Chart(pair_df[pair_df["selected_design"]])
                                                .mark_circle(size=165, fillOpacity=0.15, stroke="#111827", strokeWidth=3)
                                                .encode(
                                                    x=alt.X("run_a_rank:Q"),
                                                    y=alt.Y("run_b_rank:Q"),
                                                    tooltip=["binder_id:N", "run_a_rank:Q", "run_b_rank:Q"],
                                                )
                                            )
                                            pair_chart = (
                                                (pair_base + pair_highlight)
                                                .properties(title=f"Pairwise rank agreement (n={len(pair_df)})", height=260)
                                                .interactive()
                                            )
                                            pair_event = st.altair_chart(
                                                pair_chart,
                                                width="stretch",
                                                key=f"refolding_compare_pair_select_{preset_token}_{_safe_sort_token(engine)}",
                                                on_select="rerun",
                                                selection_mode="selected_design",
                                            )
                                            clicked_binder = _selected_binder_from_altair_event(pair_event)
                                            clicked_binder_key = str(clicked_binder or "").lower()
                                            if (
                                                clicked_binder_key
                                                and clicked_binder_key in engine_design_options
                                                and clicked_binder_key != str(selected_engine_design).lower()
                                            ):
                                                st.session_state[selected_state_key] = clicked_binder_key
                                                st.rerun()
                                            if len(selected_runs) == 2 and engine in STRUCTURE_ENGINE_BY_LABEL:
                                                overlay_height = st.slider(
                                                    f"{engine} overlay height",
                                                    min_value=280,
                                                    max_value=760,
                                                    value=420,
                                                    step=40,
                                                    key=f"refolding_compare_engine_overlay_height_{prediction_geometry_key}",
                                                )
                                                overlay_entries, overlay_status_rows, selected_binder_rmsd = _prediction_overlay_for_design(
                                                    selected_runs.iloc[0].to_dict(),
                                                    selected_runs.iloc[1].to_dict(),
                                                    engine,
                                                    str(selected_engine_design),
                                                )
                                                if overlay_entries:
                                                    if selected_binder_rmsd is not None:
                                                        st.caption(f"Selected design predicted binder CA RMSD: {selected_binder_rmsd:.2f} A")
                                                    components.html(
                                                        _py3dmol_overlay_html(
                                                            overlay_entries,
                                                            div_id=(
                                                                f"refolding-compare-engine-overlay-{prediction_geometry_key}-"
                                                                f"{_safe_sort_token(selected_engine_design)}"
                                                            ),
                                                            height=int(overlay_height),
                                                            show_targets=True,
                                                            show_binders=True,
                                                        ),
                                                        height=int(overlay_height) + 55,
                                                    )
                                                else:
                                                    st.warning(f"No {engine} predicted structures could be overlaid for the selected design.")
                                                with st.expander(f"{engine} selected overlay files and alignment", expanded=False):
                                                    st.dataframe(pd.DataFrame(overlay_status_rows), width="stretch", hide_index=True)
                                    with st.expander(f"{engine} rank table", expanded=False):
                                        pivot = engine_compare.pivot_table(
                                            index=["binder_id", "input_rank"],
                                            columns="run",
                                            values="output_rank",
                                            aggfunc="min",
                                        ).reset_index()
                                        st.dataframe(
                                            pivot.sort_values("input_rank", na_position="last"),
                                            width="stretch",
                                            hide_index=True,
                                        )
                            if all_comparison_rows:
                                if len(selected_runs) == 2:
                                    st.markdown("**Target-Aligned Structure Comparison**")
                                    st.caption(
                                        "Overlay two selected refolding runs for the same design. "
                                        "Every prediction is superposed onto the same staged target PDB from run 1."
                                    )
                                    overlay_build_key = f"refolding_compare_overlay_ready_{preset_token}"
                                    overlay_build_clicked = st.button(
                                        "Build structure overlay",
                                        key=f"refolding_compare_build_overlay_{preset_token}",
                                        help="Parses and aligns prediction structures for the selected engines. This can take a moment.",
                                    )
                                    if overlay_build_clicked:
                                        st.session_state[overlay_build_key] = True
                                    first_rank_map = input_rank_maps.get(str(selected_runs.iloc[0].get("run_id"))) or {}
                                    design_options = sorted(
                                        common_ids,
                                        key=lambda design: first_rank_map.get(str(design).lower(), 10**9),
                                    )
                                    selected_overlay_design = st.selectbox(
                                        "Design for structural overlay",
                                        design_options,
                                        format_func=lambda design: f"#{first_rank_map.get(str(design).lower(), '?')} | {design}",
                                        key=f"refolding_compare_overlay_design_{preset_token}",
                                    )
                                    overlay_engine_options = [
                                        engine
                                        for engine in selected_engines
                                        if engine in STRUCTURE_ENGINE_BY_LABEL
                                    ]
                                    selected_overlay_engines = st.multiselect(
                                        "Overlay engines",
                                        overlay_engine_options,
                                        default=overlay_engine_options,
                                        key=f"refolding_compare_overlay_engines_{preset_token}",
                                    )
                                    overlay_cols = st.columns([1, 1, 2])
                                    show_overlay_targets = overlay_cols[0].checkbox(
                                        "Show targets",
                                        value=True,
                                        key=f"refolding_compare_overlay_targets_{preset_token}",
                                    )
                                    show_overlay_binders = overlay_cols[1].checkbox(
                                        "Show binders",
                                        value=True,
                                        key=f"refolding_compare_overlay_binders_{preset_token}",
                                    )
                                    overlay_height = overlay_cols[2].slider(
                                        "Matrix height",
                                        min_value=520,
                                        max_value=1200,
                                        value=760,
                                        step=40,
                                        key=f"refolding_compare_overlay_height_{preset_token}",
                                    )
                                    run_records = selected_runs.to_dict(orient="records")
                                    run_a = run_records[0]
                                    run_b = run_records[1]
                                    overlay_panels: list[dict[str, object]] = []
                                    overlay_status_rows: list[dict[str, object]] = []
                                    if st.session_state.get(overlay_build_key):
                                        reference_path, reference_chains, reference_label = _target_reference_for_design(
                                            Path(str(run_a.get("run_dir") or "")),
                                            str(selected_overlay_design),
                                        )
                                        if reference_path is None or not reference_chains:
                                            st.warning("No staged target reference was found for the selected design in run 1.")
                                        else:
                                            st.caption(
                                                f"Alignment reference: {reference_label} `{reference_path.name}` "
                                                f"chains {','.join(reference_chains)}."
                                            )
                                            try:
                                                reference_structure = _parse_structure_file(reference_path)
                                            except Exception as exc:
                                                st.warning(f"Could not parse alignment reference target: {exc}")
                                                reference_structure = None
                                            if reference_structure is not None:
                                                with st.spinner("Building target-aligned structure overlay..."):
                                                    for overlay_engine in selected_overlay_engines:
                                                        aligned_entries: list[dict[str, object]] = []
                                                        engine_status_rows: list[dict[str, object]] = []
                                                        for run_index, run_row in enumerate([run_a, run_b]):
                                                            run_dir = Path(str(run_row.get("run_dir") or ""))
                                                            path, status = _engine_structure_for_design(
                                                                run_dir,
                                                                overlay_engine,
                                                                str(selected_overlay_design),
                                                            )
                                                            target_chains = _run_target_chains(run_dir)
                                                            row_status = {
                                                                "engine": overlay_engine,
                                                                "run": f"{run_row.get('job_code')} | {run_row.get('target_id')}",
                                                                "target_source": run_row.get("target_source"),
                                                                "target_chains": ",".join(target_chains),
                                                                "reference_chains": ",".join(reference_chains),
                                                                "status": status,
                                                                "path": str(path or ""),
                                                            }
                                                            if path is not None:
                                                                aligned_text, rmsd, note = _aligned_structure_text(
                                                                    path,
                                                                    reference_structure,
                                                                    reference_chains,
                                                                    target_chains,
                                                                )
                                                                row_status["alignment"] = note
                                                                row_status["target_ca_rmsd"] = rmsd
                                                                if aligned_text:
                                                                    aligned_entries.append(
                                                                        {
                                                                            "pdb": aligned_text,
                                                                            "targetChains": reference_chains,
                                                                            "run_index": run_index,
                                                                        }
                                                                    )
                                                            engine_status_rows.append(row_status)
                                                        if len(aligned_entries) == 2:
                                                            try:
                                                                aligned_a = _pdb_text_to_structure(
                                                                    str(aligned_entries[0]["pdb"]),
                                                                    f"{_safe_sort_token(overlay_engine)}_run_a",
                                                                )
                                                                aligned_b = _pdb_text_to_structure(
                                                                    str(aligned_entries[1]["pdb"]),
                                                                    f"{_safe_sort_token(overlay_engine)}_run_b",
                                                                )
                                                                binder_rmsd = _atom_rmsd(
                                                                    list(
                                                                        zip(
                                                                            _target_ca_atoms(aligned_a, ["A"]),
                                                                            _target_ca_atoms(aligned_b, ["A"]),
                                                                        )
                                                                    )
                                                                )
                                                            except Exception:
                                                                binder_rmsd = None
                                                            for status_row in engine_status_rows:
                                                                status_row["predicted_binder_ca_rmsd"] = binder_rmsd
                                                            overlay_panels.append(
                                                                {
                                                                    "engine": overlay_engine,
                                                                    "runA": {
                                                                        "pdb": aligned_entries[0]["pdb"],
                                                                        "targetChains": reference_chains,
                                                                    },
                                                                    "runB": {
                                                                        "pdb": aligned_entries[1]["pdb"],
                                                                        "targetChains": reference_chains,
                                                                    },
                                                                }
                                                            )
                                                        overlay_status_rows.extend(engine_status_rows)
                                        if overlay_panels:
                                            components.html(
                                                _py3dmol_overlay_grid_html(
                                                    overlay_panels,
                                                    div_id=f"refolding-overlay-grid-{_safe_sort_token(str(selected_overlay_design))}-{preset_token}",
                                                    height=int(overlay_height),
                                                    show_targets=show_overlay_targets,
                                                    show_binders=show_overlay_binders,
                                                ),
                                                height=int(overlay_height) + 70,
                                            )
                                        else:
                                            st.warning("No selected engine had two structures that could be target-aligned for this design.")
                                        with st.expander("Structure overlay files and alignment", expanded=False):
                                            st.dataframe(pd.DataFrame(overlay_status_rows), width="stretch", hide_index=True)
                                    else:
                                        st.caption(
                                            "Structure overlays are not built until requested; this keeps the rank/correlation view responsive."
                                        )
                                st.markdown("**Largest Rank Shifts Across Selected Engine Panels**")
                                shifts = pd.DataFrame(all_comparison_rows).dropna(subset=["rank_delta"]).copy()
                                shifts["absolute_shift"] = shifts["rank_delta"].abs()
                                shift_cols = [
                                    "engine",
                                    "feature",
                                    "run",
                                    "target_id",
                                    "binder_id",
                                    "input_rank",
                                    "output_rank",
                                    "rank_delta",
                                    "score",
                                ]
                                st.dataframe(
                                    shifts.sort_values("absolute_shift", ascending=False)[shift_cols].head(150),
                                    width="stretch",
                                    hide_index=True,
                                )

with legacy_tab:
    st.subheader("Legacy Validation Controls")
    st.caption("Older single-path validation entry points are kept here while the benchmark-style evaluator becomes the main path.")

    st.markdown("**ESMFold2 Complex Validation**")
    esm_mode = st.segmented_control(
        "ESMFold2 mode",
        ["sequence", "initial_guess"],
        selection_mode="single",
        default="sequence",
        format_func={"sequence": "Sequence only", "initial_guess": "Initial guess"}.get,
        key="esmfold2_validation_mode",
    )
    esm_cols = st.columns(5)
    with esm_cols[0]:
        esm_max_candidates = st.number_input("Max candidates", min_value=1, max_value=1000, value=min(20, max(1, int(source["candidate_count"]))), step=1, key="esmfold2_validation_max_candidates")
    with esm_cols[1]:
        esm_sampling_steps = st.number_input("Sampling steps", min_value=1, max_value=256, value=32, step=1, key="esmfold2_validation_sampling_steps")
    with esm_cols[2]:
        esm_loops = st.number_input("Recycling loops", min_value=1, max_value=16, value=3, step=1, key="esmfold2_validation_loops")
    with esm_cols[3]:
        esm_seed = st.number_input("Seed", min_value=0, max_value=999999, value=0, step=1, key="esmfold2_validation_seed")
    with esm_cols[4]:
        esm_contact_cutoff = st.number_input("Contact cutoff", min_value=2.0, max_value=20.0, value=8.0, step=0.5, key="esmfold2_validation_contact_cutoff")
    esm_use_target_msa = st.checkbox(
        "Use ESMFold2 target MSAs",
        value=True,
        key="esmfold2_validation_use_target_msa",
        help="Pass cached per-target-chain A3M files into ESMFold2 ProteinInput.msa when they match the target sequence.",
    )
    esm_device = st.segmented_control(
        "ESMFold2 device",
        ["auto", "cuda"],
        selection_mode="single",
        default="auto",
        format_func={"auto": "Auto (scheduled GPU)", "cuda": "CUDA"}.get,
        key="esmfold2_validation_device",
    )
    st.caption("Biohub ESMFold2 requires CUDA. The scheduler assigns its GPU and limits the container's CPU cores.")
    esm_validation_cpu_cores = cpu_run_panel(key="esmfold2_validation", default=4)
    if st.button("Run ESMFold2 validation", type="primary"):
        try:
            with st.spinner("Queueing ESMFold2 complex validation..."):
                run_dir = queue_esmfold2_complex_validation(
                    source_run_dir=Path(str(source["run_dir"])),
                    candidates_jsonl=Path(str(source["candidates_jsonl"])),
                    max_candidates=int(esm_max_candidates),
                    num_loops=int(esm_loops),
                    num_sampling_steps=int(esm_sampling_steps),
                    seed=int(esm_seed),
                    device=str(esm_device or "auto"),
                    contact_cutoff=float(esm_contact_cutoff),
                    use_initial_guess=str(esm_mode or "sequence") == "initial_guess",
                    use_target_msa=bool(esm_use_target_msa),
                    queue_cpu_cores=esm_validation_cpu_cores,
                )
            st.success("ESMFold2 validation queued. It will continue if you close Streamlit.")
            show_pipeline_links(run_dir, [run_dir])
        except Exception as exc:
            st.error(str(exc))

    st.markdown("**Run Full Refolding / Validation**")
    monomer_tool = st.segmented_control(
        "Monomer tool",
        ["af2_monomer", "boltz2_monomer", "esmfold"],
        selection_mode="single",
        default="boltz2_monomer",
        format_func={"af2_monomer": "AF2 monomer", "boltz2_monomer": "Boltz2 monomer", "esmfold": "ESMFold"}.get,
    )
    min_plddt = st.number_input("Minimum monomer pLDDT", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    complex_tool = st.segmented_control(
        "Complex tool",
        ["af2_initial_guess", "boltz2_initial_guess", "esmfold2_complex_validation", "esmfold2_initial_guess_validation"],
        selection_mode="single",
        default="af2_initial_guess",
        format_func={
            "af2_initial_guess": "AF2 initial guess",
            "boltz2_initial_guess": "Boltz2 initial guess",
            "esmfold2_complex_validation": "ESMFold2 validation",
            "esmfold2_initial_guess_validation": "ESMFold2 initial guess",
        }.get,
    )
    selected_complex_tool = str(complex_tool or "af2_initial_guess")
    if selected_complex_tool == "boltz2_initial_guess":
        template_mode = st.segmented_control("Boltz2 template mode", ["target_template", "no_template"], selection_mode="single", default="target_template", format_func={"target_template": "Target template", "no_template": "No template"}.get)
        multimer = True
    elif selected_complex_tool in {"esmfold2_complex_validation", "esmfold2_initial_guess_validation"}:
        template_mode = "target_template"
        multimer = True
    else:
        af2_options = {
            "af2_model_1_multimer_tt_3rec": ("AF2 multimer, target template", "target_template", True, 3),
            "af2_model_1_ptm_tt_3rec": ("AF2 monomer model, target template", "target_template", False, 3),
            "af2_model_1_multimer_tbt_3rec": ("AF2 multimer, target + binder templates", "target_binder_template", True, 3),
            "af2_model_1_multimer_ct_3rec": ("AF2 multimer, complex template", "complex_template", True, 3),
        }
        af2_choice = st.selectbox("AF2 complex refolding model", list(af2_options), format_func=lambda key: af2_options[key][0], key="full_validation_af2_model")
        _label, template_mode, multimer, _recycles = af2_options[af2_choice]
    analysis_cols = st.columns(4)
    with analysis_cols[0]:
        keep_top_n = st.number_input("Keep top", min_value=1, max_value=10000, value=100, step=10, key="full_validation_keep_top")
    with analysis_cols[1]:
        min_binder_plddt = st.number_input("Min binder pLDDT", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="full_validation_plddt")
    with analysis_cols[2]:
        max_ipae = st.number_input("Max iPAE", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="full_validation_ipae")
    with analysis_cols[3]:
        max_binder_rmsd = st.number_input("Max binder RMSD", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="full_validation_rmsd")
    steps = [
        {"module": "monomer_refolding", "tool": str(monomer_tool or "boltz2_monomer"), "params": {"min_plddt": float(min_plddt)}},
        {
            "module": "complex_refolding",
            "tool": selected_complex_tool,
            "params": {
                "require_monomer_success": True,
                "template_mode": str(template_mode or "target_template"),
                "multimer": bool(multimer),
                "num_recycles": 3,
                "max_candidates": int(esm_max_candidates),
                "num_sampling_steps": int(esm_sampling_steps),
                "seed": int(esm_seed),
                "device": str(esm_device or "auto"),
                "contact_cutoff": float(esm_contact_cutoff),
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
                    "min_ipsae": 0.0,
                    "max_ipae": float(max_ipae),
                    "max_ipde": 20.0,
                    "max_binder_rmsd": float(max_binder_rmsd),
                },
            },
        },
    ]
    validation_cpu_cores = cpu_run_panel(key="full_validation", default=4)
    if st.button("Run full validation", type="primary"):
        try:
            with st.spinner("Queueing monomer refolding, complex refolding, and analysis..."):
                run_dir = queue_validation_sequence(
                    campaign_name=f"Validation from {source['job_code']} {source['tool']}",
                    initial_source=source_from_run_dir(Path(str(source["run_dir"]))),
                    steps=steps,
                    cpu_cores=int(validation_cpu_cores),
                    gpu_device="0",
                )
            st.success("Full validation queued. Its refolding and analysis steps will continue if you close Streamlit.")
            show_pipeline_links(None, [run_dir])
        except Exception as exc:
            st.error(str(exc))
