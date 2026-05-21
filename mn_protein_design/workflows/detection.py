from __future__ import annotations

import glob
from pathlib import Path
import shutil
import subprocess

from mn_protein_design.core.artifacts import Artifact, artifact_path
from mn_protein_design.core.jobs import collect_jobs, create_job, finish_job, read_json, update_status, write_json
from mn_protein_design.core.manifests import load_manifest
from mn_protein_design.core.structures import filter_pdb_text, pdb_summary, sanitize_pdb_for_surface_tools
from mn_protein_design.runtime import PROJECT_DIR


DETECTION_GROUP = "detection"
HOTSPOT_GROUP = "hotspot-detection"


def completed_target_jobs() -> list[dict]:
    rows: list[dict] = []

    for row in collect_jobs("target-prep"):
        if row.get("status") != "completed":
            continue
        run_dir = Path(row["run_dir"])
        result = read_json(run_dir / "result.json")
        target = (result.get("outputs") or {}).get("target") or {}
        target_pdb = run_dir / result.get("downstream_artifacts", {}).get("target_trimmed_pdb", "artifacts/target_trimmed.pdb")
        if not target_pdb.exists():
            target_pdb = run_dir / result.get("downstream_artifacts", {}).get("target_clean_pdb", "artifacts/target_clean.pdb")
        if not target_pdb.exists():
            continue
        rows.append(
            {
                **row,
                "target_name": target.get("target_name") or row["job_code"],
                "target_pdb": str(target_pdb),
                "chains": target.get("chains") or [],
                "residue_count": target.get("residue_count"),
            }
        )

    for row in collect_jobs("target-crop"):
        if row.get("status") != "completed":
            continue
        run_dir = Path(row["run_dir"])
        result = read_json(run_dir / "result.json")
        crop = (result.get("outputs") or {}).get("crop") or {}
        target_pdb = run_dir / result.get("downstream_artifacts", {}).get("target_cropped_pdb", "artifacts/target_cropped.pdb")
        if not target_pdb.exists():
            continue
        summary = pdb_summary(target_pdb.read_text(errors="ignore"))
        chains = [chain["chain_id"] for chain in summary["chains"]]
        source_name = crop.get("target_name") or (result.get("inputs") or {}).get("target_name") or row["job_code"]
        rows.append(
            {
                **row,
                "target_name": f"{source_name} crop",
                "target_pdb": str(target_pdb),
                "chains": chains,
                "residue_count": sum(chain["residue_count"] for chain in summary["chains"]),
            }
        )

    rows.sort(key=lambda row: row.get("updated_at") or row.get("created_at") or "", reverse=True)
    return rows


def detection_jobs_for_target(target_pdb: Path) -> list[dict]:
    target_path = str(target_pdb)
    rows: list[dict] = []
    for group in [DETECTION_GROUP, HOTSPOT_GROUP]:
        for row in collect_jobs(group):
            run_dir = Path(row["run_dir"])
            payload = read_json(run_dir / "input.json")
            inputs = payload.get("inputs") or {}
            if inputs.get("target_pdb") != target_path:
                continue
            result = read_json(run_dir / "result.json")
            rows.append(
                {
                    **row,
                    "task_group": group,
                    "success": result.get("success"),
                    "artifact_count": (result.get("metrics") or {}).get("artifact_count", ""),
                }
            )
    return rows


def target_label(row: dict) -> str:
    chains = ",".join(row.get("chains") or [])
    return f"{row['target_name']} ({row['job_code']}, chains {chains or 'unknown'})"


def _run_shell_steps(run_dir: Path, steps: list[dict]) -> int:
    write_json(run_dir / "command.json", {"mode": "docker", "steps": steps})
    update_status(run_dir, "running")
    with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
        for step in steps:
            stdout.write(f"$ {' '.join(step['command'])}\n")
            stdout.flush()
            proc = subprocess.run(step["command"], stdout=stdout, stderr=stderr, check=False)
            if proc.returncode != 0:
                return int(proc.returncode)
    return 0


def _docker_base(image: str, run_dir: Path, gpu: bool = False) -> list[str]:
    command = ["docker", "run", "--rm"]
    if gpu:
        command.extend(["--gpus", "all"])
    command.extend(["-v", f"{run_dir}:/work"])
    return command + [image]


def _collect_artifacts(run_dir: Path, root: Path, patterns: list[tuple[str, str]]) -> list[dict]:
    artifacts: list[dict] = []
    for pattern, artifact_type in patterns:
        for path_text in glob.glob(str(root / pattern), recursive=True):
            path = Path(path_text)
            if path.is_file():
                artifacts.append(Artifact(path.stem, path, artifact_type).to_json(run_dir))
    return artifacts


def run_scannet(target_pdb: Path, chain_ids: list[str], mode: str = "interface", use_msa: bool = False) -> Path:
    manifest = load_manifest("scannet")
    job = create_job(
        DETECTION_GROUP,
        job_type="ppi_detection",
        tool="scannet",
        inputs={"target_pdb": str(target_pdb), "chains": chain_ids},
        params={"mode": mode, "use_msa": use_msa},
    )
    input_pdb = artifact_path(job.run_dir, "input", "target.pdb")
    input_pdb.write_text(filter_pdb_text(target_pdb.read_text(errors="ignore"), keep_chains=set(chain_ids) or None))
    out_dir = artifact_path(job.run_dir, "scannet")
    out_dir.mkdir(parents=True, exist_ok=True)
    query = "/work/artifacts/input/target.pdb"
    args = ["python", "predict_bindingsites.py", query, "--name", "target", "--predictions_folder", "/work/artifacts/scannet", "--mode", mode]
    if not use_msa:
        args.append("--noMSA")
    steps = [{"name": "scannet", "command": _docker_base(manifest["image"], job.run_dir, gpu=False) + args}]
    rc = _run_shell_steps(job.run_dir, steps)
    artifacts = _collect_artifacts(job.run_dir, out_dir, [("**/*.csv", "residue_score_table"), ("**/*.pdb", "pdb"), ("**/*.cxc", "chimera_script"), ("**/*.py", "script")])
    finish_job(
        job.run_dir,
        rc == 0,
        {
            "outputs": {"artifacts": artifacts},
            "metrics": {"return_code": rc, "artifact_count": len(artifacts)},
            "downstream_artifacts": {"residue_scores": [a["path"] for a in artifacts if a["type"] == "residue_score_table"]},
        },
    )
    return job.run_dir


def run_surf2spot(target_pdb: Path) -> Path:
    manifest = load_manifest("surf2spot")
    job = create_job(
        HOTSPOT_GROUP,
        job_type="hotspot_detection",
        tool="surf2spot",
        inputs={"target_pdb": str(target_pdb)},
        params={"mode": "HS"},
    )
    input_pdb = artifact_path(job.run_dir, "surf2spot", "input", "target.pdb")
    sanitized_text, sanitize_metrics = sanitize_pdb_for_surface_tools(target_pdb.read_text(errors="ignore"))
    input_pdb.write_text(sanitized_text)
    base = ["docker", "run", "--rm", "--gpus", "all", "-v", f"{job.run_dir}/artifacts/surf2spot:/work"]
    chain_ref = "/mnt/db/reference_files/surf2spot/chainsaw/saved_models"
    emb_ref = "/mnt/db/reference_files/surf2spot/model_emb/prot_t5_xl_half_uniref50-enc"
    common_mounts = ["-v", f"{chain_ref}:/opt/Surf2Spot/Surf2Spot/data/chainsaw/saved_models:ro"]
    emb_mount = ["-v", f"{emb_ref}:/opt/Surf2Spot/Surf2Spot/data/model_emb/prot_t5_xl_half_uniref50-enc:ro"]
    image = manifest["image"]
    steps = [
        {"name": "HS-preprocess", "command": base + common_mounts + [image, "Surf2Spot", "HS-preprocess", "-i", "/work/input", "-o", "/work/preprocess"]},
        {"name": "HS-craft", "command": base + common_mounts + emb_mount + [image, "Surf2Spot", "HS-craft", "-i", "/work/preprocess"]},
        {"name": "HS-predict", "command": base + common_mounts + emb_mount + [image, "Surf2Spot", "HS-predict", "-i", "/work/preprocess", "-o", "/work/predict", "--model", "/opt/Surf2Spot/model/HS/model.pt"]},
        {"name": "HS-draw", "command": base + common_mounts + emb_mount + [image, "Surf2Spot", "HS-draw", "-i", "/work/preprocess", "-o", "/work/predict"]},
    ]
    rc = _run_shell_steps(job.run_dir, steps)
    root = job.run_dir / "artifacts" / "surf2spot"
    artifacts = _collect_artifacts(job.run_dir, root, [("predict/*.csv", "hotspot_table"), ("predict/*_pred.ply", "surface_mesh"), ("predict/*.pse", "pymol_session")])
    finish_job(
        job.run_dir,
        rc == 0,
        {
            "outputs": {"artifacts": artifacts},
            "metrics": {"return_code": rc, "artifact_count": len(artifacts), **sanitize_metrics},
            "downstream_artifacts": {
                "hotspot_tables": [a["path"] for a in artifacts if a["type"] == "hotspot_table"],
                "hotspot_surfaces": [a["path"] for a in artifacts if a["type"] == "surface_mesh"],
            },
        },
    )
    return job.run_dir


def run_masif_seed(target_pdb: Path, chain_id: str) -> Path:
    manifest = load_manifest("masif_seed")
    job = create_job(
        DETECTION_GROUP,
        job_type="surface_ppi_detection",
        tool="masif_seed",
        inputs={"target_pdb": str(target_pdb), "chain_id": chain_id},
        params={"mode": "target_site_surface"},
    )
    input_pdb = artifact_path(job.run_dir, "masif_seed", "input", "target.pdb")
    shutil.copyfile(target_pdb, input_pdb)
    target_id = f"TGT1_{chain_id}"
    repo = PROJECT_DIR / "tools_to_implement" / "masif_seed"
    script = (
        "set -euxo pipefail; "
        "cd /work/masif_seed_search/data/masif_targets; "
        f"./data_prepare_one.sh --file /job/artifacts/masif_seed/input/target.pdb {target_id}; "
        f"./predict_site.sh {target_id}; "
        f"./color_site.sh {target_id}; "
        f"./compute_descriptors.sh {target_id}; "
        "mkdir -p /job/artifacts/masif_seed/pred_surfaces /job/artifacts/masif_seed/pred_data /job/artifacts/masif_seed/target_run; "
        f"cp -av output/all_feat_3l/pred_surfaces/{target_id}.ply /job/artifacts/masif_seed/pred_surfaces/ 2>/dev/null || true; "
        f"cp -av output/all_feat_3l/pred_data/pred_{target_id}.npy /job/artifacts/masif_seed/pred_data/ 2>/dev/null || true; "
        f"cp -a targets/{target_id} /job/artifacts/masif_seed/target_run/ 2>/dev/null || true"
    )
    steps = [
        {
            "name": "masif-target-site",
            "command": [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-v",
                f"{repo}:/work",
                "-v",
                f"{job.run_dir}:/job",
                "-w",
                "/work",
                manifest["image"],
                "bash",
                "-lc",
                script,
            ],
        }
    ]
    rc = _run_shell_steps(job.run_dir, steps)
    root = job.run_dir / "artifacts" / "masif_seed"
    artifacts = _collect_artifacts(job.run_dir, root, [("pred_surfaces/*.ply", "surface_mesh"), ("pred_data/*.npy", "masif_site_prediction"), ("target_run/**/*", "masif_output")])
    finish_job(
        job.run_dir,
        rc == 0,
        {
            "outputs": {"artifacts": artifacts, "masif_target_id": target_id},
            "metrics": {"return_code": rc, "artifact_count": len(artifacts)},
            "downstream_artifacts": {
                "predicted_surfaces": [a["path"] for a in artifacts if a["type"] == "surface_mesh"],
                "site_predictions": [a["path"] for a in artifacts if a["type"] == "masif_site_prediction"],
            },
        },
    )
    return job.run_dir


def chains_for_pdb(path: Path) -> list[str]:
    return [chain["chain_id"] for chain in pdb_summary(path.read_text(errors="ignore"))["chains"]]
