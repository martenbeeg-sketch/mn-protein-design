from __future__ import annotations

from pathlib import Path
import shutil

from mn_protein_design.core.artifacts import Artifact, artifact_path
from mn_protein_design.core.jobs import create_job, finish_job, update_status, write_json
from mn_protein_design.core.residue_selection import parse_selection
from mn_protein_design.core.structures import (
    clean_pdb,
    map_modified_residues_to_standard,
    pdb_chains,
    replace_nonstandard_residues_with_pdbfixer,
)


TASK_GROUP = "target-prep"


def prepare_target(
    source_pdb: Path,
    target_name: str,
    keep_chains: list[str] | None = None,
    residue_range_text: str = "",
    remove_waters: bool = True,
    remove_hetero: bool = False,
    map_known_modified_residues: bool = False,
    replace_nonstandard_residues: bool = False,
) -> Path:
    source_pdb = source_pdb.expanduser().resolve()
    if not source_pdb.exists():
        raise FileNotFoundError(f"Input PDB does not exist: {source_pdb}")
    if source_pdb.suffix.lower() not in {".pdb", ".ent"}:
        raise ValueError("Target preparation currently expects a PDB file.")

    keep_chain_set = {c.strip() for c in keep_chains or [] if c.strip()} or None
    residue_selections = parse_selection(residue_range_text)
    inputs = {
        "source_pdb": str(source_pdb),
        "target_name": target_name,
        "keep_chains": sorted(keep_chain_set or []),
        "residue_selections": [s.to_json() for s in residue_selections],
    }
    params = {
        "remove_waters": remove_waters,
        "remove_hetero": remove_hetero,
        "map_known_modified_residues": map_known_modified_residues,
        "replace_nonstandard_residues": replace_nonstandard_residues,
    }
    job = create_job(TASK_GROUP, job_type="target_preparation", tool="internal_pdb_cleaner", inputs=inputs, params=params)
    update_status(job.run_dir, "running")
    write_json(
        job.run_dir / "command.json",
        {"mode": "internal", "command": ["internal_pdb_cleaner"], "source_pdb": str(source_pdb)},
    )

    try:
        raw_target = artifact_path(job.run_dir, "target_input.pdb")
        mapped_target = artifact_path(job.run_dir, "target_mapped.pdb")
        repaired_target = artifact_path(job.run_dir, "target_repaired.pdb")
        cleaned_target = artifact_path(job.run_dir, "target_clean.pdb")
        trimmed_target = artifact_path(job.run_dir, "target_trimmed.pdb")
        target_json = artifact_path(job.run_dir, "target.json")
        shutil.copyfile(source_pdb, raw_target)
        mapping_metrics = {}
        repair_metrics = {}
        clean_input = raw_target
        current_text = raw_target.read_text(errors="ignore")
        if map_known_modified_residues:
            mapped_text, mapping_metrics = map_modified_residues_to_standard(current_text)
            mapped_target.write_text(mapped_text)
            clean_input = mapped_target
            current_text = mapped_text
        if replace_nonstandard_residues:
            repaired_text, repair_metrics = replace_nonstandard_residues_with_pdbfixer(current_text)
            repaired_target.write_text(repaired_text)
            clean_input = repaired_target
        stats = clean_pdb(
            clean_input,
            cleaned_target,
            keep_chains=keep_chain_set,
            remove_waters=remove_waters,
            remove_hetero=remove_hetero,
            residue_selections=residue_selections,
        )
        shutil.copyfile(cleaned_target, trimmed_target)
        target_payload = {
            "target_name": target_name,
            "source_pdb": str(source_pdb),
            "chains": stats["chains"],
            "residue_count": stats["residue_count"],
            "artifacts": {
                "target_clean": "artifacts/target_clean.pdb",
                "target_trimmed": "artifacts/target_trimmed.pdb",
            },
        }
        if map_known_modified_residues:
            target_payload["artifacts"]["target_mapped"] = "artifacts/target_mapped.pdb"
        if replace_nonstandard_residues:
            target_payload["artifacts"]["target_repaired"] = "artifacts/target_repaired.pdb"
        write_json(target_json, target_payload)
        artifacts = [
            Artifact("target_input", raw_target, "pdb", "Original uploaded PDB").to_json(job.run_dir),
            *(
                [
                    Artifact(
                        "target_mapped",
                        mapped_target,
                        "pdb",
                        "Known modified residues mapped to canonical amino acids",
                    ).to_json(job.run_dir)
                ]
                if map_known_modified_residues
                else []
            ),
            *(
                [Artifact("target_repaired", repaired_target, "pdb", "PDBFixer output with nonstandard residues replaced").to_json(job.run_dir)]
                if replace_nonstandard_residues
                else []
            ),
            Artifact("target_clean", cleaned_target, "pdb", "Cleaned target PDB").to_json(job.run_dir),
            Artifact("target_trimmed", trimmed_target, "pdb", "Chain-trimmed target PDB").to_json(job.run_dir),
            Artifact("target_json", target_json, "target_manifest", "Prepared target manifest").to_json(job.run_dir),
        ]
        finish_job(
            job.run_dir,
            True,
            {
                "outputs": {"artifacts": artifacts, "target": target_payload},
                "metrics": {**stats, **mapping_metrics, **repair_metrics},
                "downstream_artifacts": {
                    "target_clean_pdb": "artifacts/target_clean.pdb",
                    "target_trimmed_pdb": "artifacts/target_trimmed.pdb",
                    **({"target_mapped_pdb": "artifacts/target_mapped.pdb"} if map_known_modified_residues else {}),
                    **({"target_repaired_pdb": "artifacts/target_repaired.pdb"} if replace_nonstandard_residues else {}),
                    "target_json": "artifacts/target.json",
                },
            },
        )
    except Exception as exc:
        (job.run_dir / "stderr.log").write_text(f"{type(exc).__name__}: {exc}\n")
        finish_job(job.run_dir, False, {"outputs": {}, "metrics": {}, "error": str(exc)})
        raise
    return job.run_dir


def inspect_pdb(path: Path) -> dict:
    return {"chains": pdb_chains(path)}
