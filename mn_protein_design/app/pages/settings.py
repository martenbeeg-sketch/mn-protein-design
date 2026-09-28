from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from mn_protein_design.runtime import (
    DEFAULT_APP_HOME,
    DEFAULT_SHARED_REFERENCE_ROOT,
    DEFAULT_TMPDIR,
    PROJECT_DIR,
    app_home,
    reference_root,
    runs_root,
)
from mn_protein_design.workflows import benchmark as benchmark_workflow
from mn_protein_design.workflows import detection as detection_workflow
from mn_protein_design.workflows import esm_binder as esm_binder_workflow
from mn_protein_design.workflows import refolding as refolding_workflow
from mn_protein_design.workflows import target_msa as target_msa_workflow


st.title("Settings")
st.caption("Installation-specific paths, model references, MSA caches, and Docker images used by this app.")


def _path_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(Path(value).expanduser())
    except TypeError:
        return str(value)


def _path_status(value: Any) -> str:
    text = _path_text(value).strip()
    if not text:
        return "not set"
    path = Path(text).expanduser()
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "missing"


def _env_source(env_name: str | None, fallback: str = "derived/default") -> str:
    if env_name and os.getenv(env_name):
        return f"${env_name}"
    return fallback


def _path_row(name: str, value: Any, source: str, note: str = "") -> dict[str, str]:
    return {
        "setting": name,
        "path": _path_text(value),
        "status": _path_status(value),
        "source": source,
        "note": note,
    }


def _image_row(name: str, image: Any, note: str = "") -> dict[str, str]:
    return {"engine/tool": name, "docker image": str(image or ""), "note": note}


def _reference_path(*parts: str) -> Path:
    return reference_root().joinpath(*parts)


with st.expander("Runtime paths", expanded=True):
    st.caption("These are the core paths controlled by app launch environment variables.")
    general_rows = [
        _path_row("Project directory", PROJECT_DIR, "package location"),
        _path_row("App home", app_home(), _env_source("MN_PROTEIN_DESIGN_APP_HOME", f"default: {DEFAULT_APP_HOME}")),
        _path_row("Runs / data folder", runs_root(), _env_source("MN_PROTEIN_DESIGN_RUN_DIR", "derived from app home")),
        _path_row("Target artifact folder", app_home() / "workdir" / "targets", "derived from app home"),
        _path_row("Reference files root", reference_root(), _env_source("MN_PROTEIN_DESIGN_REFERENCE_DIR", f"default: {DEFAULT_SHARED_REFERENCE_ROOT}")),
        _path_row("Temporary folder", Path(os.getenv("TMPDIR", str(DEFAULT_TMPDIR))), _env_source("TMPDIR", f"default: {DEFAULT_TMPDIR}")),
    ]
    st.dataframe(pd.DataFrame(general_rows), hide_index=True, use_container_width=True)

    app_home_value = st.text_input("App home", value=str(app_home()))
    runs_value = st.text_input("Runs / data folder", value=str(runs_root()))
    reference_value = st.text_input("Reference files root", value=str(reference_root()))
    tmp_value = st.text_input("Temporary folder", value=str(Path(os.getenv("TMPDIR", str(DEFAULT_TMPDIR))).expanduser()))

    st.caption("Restart the app with these exports to apply path changes to all freshly imported modules.")
    st.code(
        "\n".join(
            [
                f"export MN_PROTEIN_DESIGN_APP_HOME={app_home_value}",
                f"export MN_PROTEIN_DESIGN_RUN_DIR={runs_value}",
                f"export MN_PROTEIN_DESIGN_REFERENCE_DIR={reference_value}",
                f"export TMPDIR={tmp_value}",
                "mn-protein-design app",
            ]
        ),
        language="bash",
    )


with st.expander("MSA and reference data", expanded=True):
    st.caption("These paths are consumed by refolding, benchmark, design, and target-MSA preparation jobs.")
    reference_rows = [
        _path_row("Boltz target MSA repository", target_msa_workflow.BOLTZ_MSA_REPOSITORY_DIR, "target_msa.BOLTZ_MSA_REPOSITORY_DIR"),
        _path_row("AlphaFast / MMseqs DB", target_msa_workflow.ALPHAFAST_DB_DIR, "target_msa.ALPHAFAST_DB_DIR"),
        _path_row("OpenFold-3 benchmark MSA prefill", refolding_workflow.OPENFOLD3_REFERENCE_MSA_PREFILL, "refolding.OPENFOLD3_REFERENCE_MSA_PREFILL"),
        _path_row("AlphaFold / ColabFold model cache", benchmark_workflow.COLABFOLD_CACHE_DIR, "benchmark.COLABFOLD_CACHE_DIR"),
        _path_row("AlphaFast AF3 weights", benchmark_workflow.ALPHAFAST_WEIGHTS_DIR, "benchmark.ALPHAFAST_WEIGHTS_DIR"),
        _path_row("Boltz model/cache root", refolding_workflow.BOLTZ_MODELS_DIR, "refolding.BOLTZ_MODELS_DIR"),
        _path_row("RF3 checkpoint", refolding_workflow.RF3_CHECKPOINT, "refolding.RF3_CHECKPOINT"),
        _path_row("OpenFold-3 checkpoint", refolding_workflow.OPENFOLD3_CHECKPOINT, "refolding.OPENFOLD3_CHECKPOINT"),
        _path_row("PXDesign / Protenix v0.5 reference", refolding_workflow.PROTENIX_REFERENCE_DIR, "refolding.PROTENIX_REFERENCE_DIR"),
        _path_row("Protenix v1/v2 reference", refolding_workflow.PROTENIX_CLI_REFERENCE_DIR, "refolding.PROTENIX_CLI_REFERENCE_DIR"),
        _path_row("Biohub ESM root", esm_binder_workflow.BIOHUB_ESM_ROOT, "esm_binder.BIOHUB_ESM_ROOT"),
        _path_row("ESMFold2 model", esm_binder_workflow.ESMFOLD2_MODEL_DIR, "esm_binder.ESMFOLD2_MODEL_DIR"),
        _path_row("ESMC model", esm_binder_workflow.ESMC_MODEL_DIR, "esm_binder.ESMC_MODEL_DIR"),
        _path_row("ESMFold2 binder-design models", esm_binder_workflow.ESMFOLD2_BINDER_MODEL_ROOT, "esm_binder.ESMFOLD2_BINDER_MODEL_ROOT"),
        _path_row("Benchmark dataset root", detection_workflow.KNOWN_BENCHMARK_ROOT, "detection.KNOWN_BENCHMARK_ROOT"),
        _path_row("PESTO model directory", _reference_path("pesto"), "reference root convention"),
        _path_row("Surf2Spot chainsaw models", _reference_path("surf2spot", "chainsaw", "saved_models"), "reference root convention"),
        _path_row("Surf2Spot ProtT5 embedding model", _reference_path("surf2spot", "model_emb", "prot_t5_xl_half_uniref50-enc"), "reference root convention"),
        _path_row("RFdiffusion model directory", _reference_path("rfdiffusion_models"), "Nextflow MN_PROTEIN_DESIGN_REFERENCE_DIR"),
        _path_row("BoltzGen cache", _reference_path("boltzgen-cache"), "design Docker mount convention"),
        _path_row("Genie3 pretrained models", _reference_path("genie3", "pretrained"), "design Docker mount convention"),
        _path_row("Protpardelle-1c reference", _reference_path("protpardelle-1c"), "design Docker mount convention"),
        _path_row("Proteina-Complexa checkpoints", _reference_path("proteina-complexa", "ckpts"), "design Docker mount convention"),
        _path_row("Proteina-Complexa HF cache", _reference_path("proteina-complexa", "hf-cache"), "design Docker mount convention"),
    ]
    st.dataframe(pd.DataFrame(reference_rows), hide_index=True, use_container_width=True)


with st.expander("Local executables and helper scripts", expanded=True):
    executable_rows = [
        _path_row("PyMOL Python", benchmark_workflow.PYMOL_PYTHON, "benchmark.PYMOL_PYTHON", "Used for PyMOL/interface geometry metrics."),
        _path_row("Boltz prepare_inputs.py", refolding_workflow.BOLTZ_PREPARE_INPUTS, "refolding.BOLTZ_PREPARE_INPUTS", "Used by Boltz-2 refolding adapter."),
        _path_row("Vendored de-novo binder scoring tools", benchmark_workflow.DE_NOVO_BINDER_SCORING_DIR, "repo-local"),
        _path_row("Vendored ESM tools", esm_binder_workflow.VENDORED_ESM_DIR, "repo-local"),
        _path_row("BoltzGen local source", refolding_workflow.BOLTZGEN_LOCAL_SOURCE, "repo-local"),
    ]
    st.dataframe(pd.DataFrame(executable_rows), hide_index=True, use_container_width=True)


with st.expander("Docker images", expanded=False):
    image_rows = [
        _image_row("AlphaFast AF3 / MMseqs", benchmark_workflow.ALPHAFAST_IMAGE),
        _image_row("ColabFold", benchmark_workflow.COLABFOLD_IMAGE),
        _image_row("AF2 initial guess / PyRosetta base", benchmark_workflow.AF2_INITIAL_GUESS_IMAGE),
        _image_row("Boltz-2", benchmark_workflow.BOLTZ2_IMAGE),
        _image_row("RF3 / Foundry", refolding_workflow.RF3_IMAGE),
        _image_row("OpenFold-3", refolding_workflow.OPENFOLD3_IMAGE),
        _image_row("Protenix v0.5 / PXDesign", refolding_workflow.PROTENIX_IMAGE),
        _image_row("Protenix v1/v2 CLI", refolding_workflow.PROTENIX_CLI_IMAGE),
        _image_row("BoltzGen", refolding_workflow.BOLTZGEN_IMAGE),
        _image_row("Biohub ESM / ESMFold2 binder design", esm_binder_workflow.ESMFOLD2_BINDER_IMAGE),
        _image_row("Structure scoring scripts", benchmark_workflow.SCORING_SCRIPTS_IMAGE),
        _image_row("PyRosetta metrics", benchmark_workflow.PYROSETTA_METRICS_IMAGE),
    ]
    st.dataframe(pd.DataFrame(image_rows), hide_index=True, use_container_width=True)


with st.expander("Reference-root migration notes", expanded=False):
    st.caption(
        "Rows here are still hard-coded or partially hard-coded in adapters. They should be migrated before a custom reference root is fully portable."
    )
    migration_rows = [
        {
            "area": "design engines",
            "files": "workflows/design.py",
            "current behavior": "Several Docker mounts still use /mnt/db/reference_files directly.",
            "preferred setting": "derive every mount from MN_PROTEIN_DESIGN_REFERENCE_DIR",
        },
        {
            "area": "detection tools",
            "files": "workflows/detection.py",
            "current behavior": "PESTO, Surf2Spot, and benchmark-data defaults use shared /mnt/db/reference_files paths.",
            "preferred setting": "derive detection references from the reference root",
        },
        {
            "area": "sequence design",
            "files": "workflows/sequence_design.py",
            "current behavior": "RF3 and LigandMPNN weight mounts use /mnt/db/reference_files directly.",
            "preferred setting": "derive model weight mounts from the reference root",
        },
        {
            "area": "refolding metrics",
            "files": "workflows/benchmark.py",
            "current behavior": "PyMOL Python points to one local conda environment.",
            "preferred setting": "make PyMOL Python configurable by environment variable",
        },
        {
            "area": "page help text",
            "files": "app/pages/*.py",
            "current behavior": "Some captions still mention /mnt/db/reference_files.",
            "preferred setting": "display the effective reference root instead",
        },
    ]
    st.dataframe(pd.DataFrame(migration_rows), hide_index=True, use_container_width=True)
