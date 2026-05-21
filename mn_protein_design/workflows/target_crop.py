from __future__ import annotations

from pathlib import Path
import shutil

from mn_protein_design.core.artifacts import Artifact, artifact_path
from mn_protein_design.core.jobs import collect_jobs, create_job, finish_job, read_json, update_status, write_json
from mn_protein_design.core.residue_selection import parse_selection
from mn_protein_design.core.structures import filter_pdb_to_residues, pdb_summary, residues_within_spheres


TASK_GROUP = "target-crop"


def _selection_rows_to_residues(selection_rows: list[dict]) -> set[tuple[str, int]]:
    residues: set[tuple[str, int]] = set()
    for row in selection_rows:
        chain = str(row.get("chainId") or row.get("chain_id") or "").strip()
        if not chain:
            continue
        for residue in row.get("residues") or []:
            try:
                residues.add((chain, int(residue)))
            except (TypeError, ValueError):
                continue
    return residues


def _manual_ranges_to_residues(manual_range_text: str) -> set[tuple[str, int]]:
    residues: set[tuple[str, int]] = set()
    for selection in parse_selection(manual_range_text):
        residues.update((selection.chain_id, residue) for residue in range(selection.start, selection.end + 1))
    return residues


def run_target_crop(
    target_pdb: Path,
    target_name: str,
    manual_range_text: str = "",
    viewer_selections: list[dict] | None = None,
    sphere_enabled: bool = False,
    sphere_diameter_angstrom: float = 12.0,
    remove_waters: bool = True,
    remove_hetero: bool = False,
) -> Path:
    target_pdb = target_pdb.expanduser().resolve()
    if not target_pdb.exists():
        raise FileNotFoundError(f"Prepared target PDB does not exist: {target_pdb}")

    manual_residues = _manual_ranges_to_residues(manual_range_text)
    clicked_residues = _selection_rows_to_residues(viewer_selections or [])
    seed_residues = manual_residues | clicked_residues
    if not seed_residues:
        raise ValueError("Select at least one residue manually or in the Mol* viewer before cropping.")

    pdb_text = target_pdb.read_text(errors="ignore")
    sphere_residues = residues_within_spheres(pdb_text, seed_residues, sphere_diameter_angstrom) if sphere_enabled else set()
    selected_residues = seed_residues | sphere_residues

    job = create_job(
        TASK_GROUP,
        job_type="target_cropping",
        tool="internal_pdb_cropper",
        inputs={
            "target_pdb": str(target_pdb),
            "target_name": target_name,
            "manual_range_text": manual_range_text,
            "viewer_selections": viewer_selections or [],
        },
        params={
            "sphere_enabled": sphere_enabled,
            "sphere_diameter_angstrom": sphere_diameter_angstrom,
            "remove_waters": remove_waters,
            "remove_hetero": remove_hetero,
        },
    )
    update_status(job.run_dir, "running")
    write_json(
        job.run_dir / "command.json",
        {"mode": "internal", "command": ["internal_pdb_cropper"], "target_pdb": str(target_pdb)},
    )

    try:
        input_pdb = artifact_path(job.run_dir, "crop_input.pdb")
        cropped_pdb = artifact_path(job.run_dir, "target_cropped.pdb")
        crop_json = artifact_path(job.run_dir, "crop.json")
        shutil.copyfile(target_pdb, input_pdb)
        cropped_text, stats = filter_pdb_to_residues(
            pdb_text,
            selected_residues,
            remove_waters=remove_waters,
            remove_hetero=remove_hetero,
        )
        if stats["residue_count"] == 0:
            raise ValueError("The crop did not contain any residues from the target structure.")
        cropped_pdb.write_text(cropped_text)
        summary = pdb_summary(cropped_text)
        selected_rows = [{"chain": chain, "residue_number": residue} for chain, residue in sorted(selected_residues)]
        crop_payload = {
            "target_name": target_name,
            "source_pdb": str(target_pdb),
            "manual_residue_count": len(manual_residues),
            "viewer_residue_count": len(clicked_residues),
            "sphere_residue_count": len(sphere_residues),
            "selected_residue_count": len(selected_residues),
            "selected_residues": selected_rows,
            "artifacts": {"target_cropped": "artifacts/target_cropped.pdb"},
        }
        write_json(crop_json, crop_payload)
        artifacts = [
            Artifact("crop_input", input_pdb, "pdb", "Prepared target used for cropping").to_json(job.run_dir),
            Artifact("target_cropped", cropped_pdb, "pdb", "Cropped target PDB").to_json(job.run_dir),
            Artifact("crop_json", crop_json, "crop_manifest", "Target crop manifest").to_json(job.run_dir),
        ]
        finish_job(
            job.run_dir,
            True,
            {
                "outputs": {"artifacts": artifacts, "crop": crop_payload},
                "metrics": {
                    **stats,
                    "manual_residue_count": len(manual_residues),
                    "viewer_residue_count": len(clicked_residues),
                    "sphere_residue_count": len(sphere_residues),
                    "selected_residue_count": len(selected_residues),
                    "cropped_chains": len(summary["chains"]),
                },
                "downstream_artifacts": {
                    "target_cropped_pdb": "artifacts/target_cropped.pdb",
                    "crop_json": "artifacts/crop.json",
                },
            },
        )
    except Exception as exc:
        (job.run_dir / "stderr.log").write_text(f"{type(exc).__name__}: {exc}\n")
        finish_job(job.run_dir, False, {"outputs": {}, "metrics": {}, "error": str(exc)})
        raise
    return job.run_dir


def completed_crop_jobs() -> list[dict]:
    rows: list[dict] = []
    for row in collect_jobs(TASK_GROUP):
        if row.get("status") != "completed":
            continue
        run_dir = Path(row["run_dir"])
        result = read_json(run_dir / "result.json")
        cropped_pdb = run_dir / result.get("downstream_artifacts", {}).get("target_cropped_pdb", "artifacts/target_cropped.pdb")
        if not cropped_pdb.exists():
            continue
        rows.append({**row, "target_pdb": str(cropped_pdb)})
    return rows
