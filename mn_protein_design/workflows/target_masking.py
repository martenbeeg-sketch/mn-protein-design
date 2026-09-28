from __future__ import annotations

from collections import defaultdict
from io import StringIO
import csv
import math
from pathlib import Path
import shutil
from typing import Any

from mn_protein_design.core.artifacts import Artifact, artifact_path
from mn_protein_design.core.jobs import create_job, finish_job, update_status, write_json
from mn_protein_design.core.structures import STANDARD_PROTEIN_RESIDUES, pdb_summary


TASK_GROUP = "target-prep"
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "PRO", "CYS"}
POLAR_RESIDUES = {"SER", "THR", "ASN", "GLN"}
CHARGED_RESIDUES = {"ASP", "GLU", "LYS", "ARG", "HIS"}


def _distance_sq(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _residue_category(resname: str) -> str:
    resname = str(resname or "").upper()
    if resname in HYDROPHOBIC_RESIDUES:
        return "hydrophobic"
    if resname in CHARGED_RESIDUES:
        return "charged"
    if resname in POLAR_RESIDUES:
        return "polar"
    if resname == "GLY":
        return "glycine"
    return "other"


def _parse_pdb_residues(pdb_text: str) -> dict[tuple[str, int, str], dict[str, Any]]:
    residues: dict[tuple[str, int, str], dict[str, Any]] = {}
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"} or len(line) < 54:
            continue
        resname = line[17:20].strip().upper()
        if resname not in STANDARD_PROTEIN_RESIDUES:
            continue
        try:
            resseq = int(line[22:26])
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        chain = line[21].strip() or "_"
        insertion = line[26].strip()
        atom_name = line[12:16].strip()
        key = (chain, resseq, insertion)
        residue = residues.setdefault(
            key,
            {
                "chain": chain,
                "resseq": resseq,
                "insertion": insertion,
                "resname": resname,
                "atoms": [],
                "sidechain_atoms": [],
                "lines": [],
            },
        )
        residue["resname"] = resname
        atom = {"name": atom_name, "xyz": xyz, "line": line}
        residue["atoms"].append(atom)
        residue["lines"].append(line)
        if atom_name not in BACKBONE_ATOMS:
            residue["sidechain_atoms"].append(atom)
    return residues


def _neighbor_sets(
    residues: dict[tuple[str, int, str], dict[str, Any]],
    *,
    cutoff_angstrom: float,
) -> dict[tuple[str, int, str], set[tuple[str, int, str]]]:
    cutoff_sq = cutoff_angstrom * cutoff_angstrom
    keys = list(residues)
    neighbors: dict[tuple[str, int, str], set[tuple[str, int, str]]] = {key: set() for key in keys}
    for i, key_a in enumerate(keys):
        sidechain_atoms = residues[key_a].get("sidechain_atoms") or residues[key_a].get("atoms") or []
        if not sidechain_atoms:
            continue
        for key_b in keys:
            if key_a == key_b:
                continue
            found = False
            for atom_a in sidechain_atoms:
                for atom_b in residues[key_b].get("atoms") or []:
                    if _distance_sq(atom_a["xyz"], atom_b["xyz"]) <= cutoff_sq:
                        neighbors[key_a].add(key_b)
                        found = True
                        break
                if found:
                    break
    return neighbors


def crop_exposure_table(
    cropped_pdb: Path,
    *,
    reference_pdb: Path | None = None,
    neighbor_cutoff_angstrom: float = 5.0,
    buried_neighbor_threshold: int = 8,
    lost_neighbor_threshold: int = 2,
    exposed_neighbor_threshold: int = 12,
) -> list[dict[str, Any]]:
    """Score residues that became artificial crop surfaces.

    The heuristic is intentionally geometry-only so it works without PyMOL/DSSP:
    count side-chain heavy-atom neighboring residues in the reference structure
    and compare that to the cropped structure. Residues that were buried in the
    reference and lost many neighbors in the crop are suggested for masking.
    """
    cropped_pdb = Path(cropped_pdb)
    reference_pdb = Path(reference_pdb) if reference_pdb else cropped_pdb
    crop_residues = _parse_pdb_residues(cropped_pdb.read_text(errors="ignore"))
    reference_residues = _parse_pdb_residues(reference_pdb.read_text(errors="ignore"))
    crop_neighbors = _neighbor_sets(crop_residues, cutoff_angstrom=neighbor_cutoff_angstrom)
    reference_neighbors = _neighbor_sets(reference_residues, cutoff_angstrom=neighbor_cutoff_angstrom)

    rows: list[dict[str, Any]] = []
    for key, residue in sorted(crop_residues.items(), key=lambda item: item[0]):
        reference_neighbor_keys = reference_neighbors.get(key, set())
        crop_neighbor_keys = crop_neighbors.get(key, set())
        removed_neighbor_keys = reference_neighbor_keys - set(crop_residues)
        full_neighbors = len(reference_neighbor_keys)
        cropped_neighbors = len(crop_neighbor_keys)
        removed_neighbors = len(removed_neighbor_keys)
        resname = str(residue.get("resname") or "").upper()
        category = _residue_category(resname)
        was_buried = full_neighbors >= buried_neighbor_threshold
        now_exposed = cropped_neighbors <= exposed_neighbor_threshold or removed_neighbors >= lost_neighbor_threshold
        crop_exposed = bool(reference_pdb != cropped_pdb and was_buried and now_exposed and removed_neighbors >= lost_neighbor_threshold)
        rows.append(
            {
                "select": crop_exposed and category == "hydrophobic",
                "chain": key[0],
                "residue_number": key[1],
                "insertion_code": key[2],
                "resname": resname,
                "category": category,
                "full_neighbors": full_neighbors,
                "cropped_neighbors": cropped_neighbors,
                "removed_neighbors": removed_neighbors,
                "was_buried": was_buried,
                "crop_exposed": crop_exposed,
                "suggested_mask": crop_exposed and category == "hydrophobic",
                "mutation": "LYS" if crop_exposed and category == "hydrophobic" else "",
            }
        )
    return rows


def parse_mutation_text(text: str, default_target: str = "LYS") -> list[dict[str, Any]]:
    """Parse manual mutations like B177K, B177:LYS, or B177."""
    aa1_to_3 = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }
    parsed: list[dict[str, Any]] = []
    default_target = str(default_target or "LYS").strip().upper()
    if len(default_target) == 1:
        default_target = aa1_to_3.get(default_target, default_target)
    for raw_item in str(text or "").replace("\n", ",").split(","):
        item = raw_item.strip().replace(" ", "")
        if not item:
            continue
        chain = item[0]
        rest = item[1:]
        number_text = ""
        suffix = ""
        for char in rest:
            if char.isdigit() or (char == "-" and not number_text):
                number_text += char
            else:
                suffix += char
        if not chain or not number_text:
            raise ValueError(f"Could not parse mutation item: {raw_item!r}")
        target = default_target
        suffix = suffix.lstrip(":=-").upper()
        if suffix:
            target = aa1_to_3.get(suffix, suffix)
        parsed.append({"chain": chain, "residue_number": int(number_text), "insertion_code": "", "mutation": target})
    return parsed


def mutation_labels(mutations: list[dict[str, Any]]) -> list[str]:
    return [f"{row['chain']}{int(row['residue_number'])}" for row in mutations]


def mutate_pdb_text(
    pdb_text: str,
    mutations: list[dict[str, Any]],
    *,
    rebuild_missing_atoms: bool = True,
) -> tuple[str, dict[str, Any]]:
    mutation_by_key = {
        (str(row["chain"]), int(row["residue_number"]), str(row.get("insertion_code") or "")): str(row["mutation"]).upper()
        for row in mutations
        if str(row.get("mutation") or "").upper() in STANDARD_PROTEIN_RESIDUES
    }
    mutated_keys: set[tuple[str, int, str]] = set()
    output: list[str] = []
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM", "ANISOU"} or len(line) < 27:
            output.append(line)
            continue
        try:
            key = (line[21].strip() or "_", int(line[22:26]), line[26].strip())
        except ValueError:
            output.append(line)
            continue
        target_resname = mutation_by_key.get(key)
        if not target_resname:
            output.append(line)
            continue
        atom_name = line[12:16].strip()
        keep_atoms = set(BACKBONE_ATOMS)
        if target_resname != "GLY":
            keep_atoms.add("CB")
        if atom_name not in keep_atoms:
            continue
        mutated_keys.add(key)
        output.append(line[:17] + f"{target_resname:>3}" + line[20:])
    if output and output[-1][:6].strip() != "END":
        output.append("END")
    mutated_text = "\n".join(output) + ("\n" if output else "")
    repair_error = ""
    repaired = False
    if rebuild_missing_atoms and mutated_keys:
        try:
            from openmm.app import PDBFile
            from pdbfixer import PDBFixer

            fixer = PDBFixer(pdbfile=StringIO(mutated_text))
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            repaired_buffer = StringIO()
            PDBFile.writeFile(fixer.topology, fixer.positions, repaired_buffer, keepIds=True)
            mutated_text = repaired_buffer.getvalue()
            repaired = True
        except Exception as exc:
            repair_error = f"{type(exc).__name__}: {exc}"
    return mutated_text, {
        "requested_mutations": len(mutation_by_key),
        "applied_mutations": len(mutated_keys),
        "rebuild_missing_atoms": bool(rebuild_missing_atoms),
        "sidechain_rebuild_success": repaired,
        "sidechain_rebuild_error": repair_error,
    }


def create_masked_target(
    source_pdb: Path,
    *,
    target_name: str,
    mutations: list[dict[str, Any]],
    reference_pdb: Path | None = None,
    mutation_source: str = "manual",
    rebuild_missing_atoms: bool = True,
) -> Path:
    source_pdb = Path(source_pdb).expanduser().resolve()
    if not source_pdb.exists():
        raise FileNotFoundError(f"Input target PDB does not exist: {source_pdb}")
    if not mutations:
        raise ValueError("Select at least one residue mutation.")
    job = create_job(
        TASK_GROUP,
        job_type="target_masking",
        tool="internal_target_masker",
        inputs={
            "source_pdb": str(source_pdb),
            "reference_pdb": str(Path(reference_pdb).expanduser().resolve()) if reference_pdb else "",
            "target_name": target_name,
            "mutation_source": mutation_source,
        },
        params={"mutations": mutations, "rebuild_missing_atoms": bool(rebuild_missing_atoms)},
    )
    update_status(job.run_dir, "running")
    write_json(job.run_dir / "command.json", {"mode": "internal", "command": ["internal_target_masker"], "source_pdb": str(source_pdb)})
    try:
        raw_target = artifact_path(job.run_dir, "target_input.pdb")
        mutated_target = artifact_path(job.run_dir, "target_masked.pdb")
        mutation_json = artifact_path(job.run_dir, "mutations.json")
        mutation_csv = artifact_path(job.run_dir, "mutations.csv")
        target_json = artifact_path(job.run_dir, "target.json")
        shutil.copyfile(source_pdb, raw_target)
        mutated_text, metrics = mutate_pdb_text(
            raw_target.read_text(errors="ignore"),
            mutations,
            rebuild_missing_atoms=rebuild_missing_atoms,
        )
        mutated_target.write_text(mutated_text)
        mutation_payload = {"mutation_source": mutation_source, "mutations": mutations, "metrics": metrics}
        write_json(mutation_json, mutation_payload)
        with mutation_csv.open("w", newline="") as handle:
            fieldnames = ["chain", "residue_number", "insertion_code", "resname", "category", "mutation", "reason"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in mutations:
                writer.writerow(row)
        summary = pdb_summary(mutated_text)
        target_payload = {
            "target_name": target_name,
            "source_pdb": str(source_pdb),
            "reference_pdb": str(reference_pdb or ""),
            "prepared_kind": "mutated",
            "chains": summary["chains"],
            "residue_count": sum(int(chain.get("residue_count") or 0) for chain in summary["chains"]),
            "mutation_count": len(mutations),
            "artifacts": {
                "target_clean": "artifacts/target_masked.pdb",
                "target_trimmed": "artifacts/target_masked.pdb",
                "target_masked": "artifacts/target_masked.pdb",
                "mutations_json": "artifacts/mutations.json",
                "mutations_csv": "artifacts/mutations.csv",
            },
        }
        write_json(target_json, target_payload)
        artifacts = [
            Artifact("target_input", raw_target, "pdb", "Input target PDB before masking").to_json(job.run_dir),
            Artifact("target_masked", mutated_target, "pdb", "Target PDB with masking/manual mutations").to_json(job.run_dir),
            Artifact("mutations_json", mutation_json, "json", "Mutation plan and mutation metrics").to_json(job.run_dir),
            Artifact("mutations_csv", mutation_csv, "csv", "Mutation plan table").to_json(job.run_dir),
            Artifact("target_json", target_json, "target_manifest", "Masked target manifest").to_json(job.run_dir),
        ]
        finish_job(
            job.run_dir,
            True,
            {
                "outputs": {"artifacts": artifacts, "target": target_payload, "mutations": mutation_payload},
                "metrics": {**metrics, **summary, "mutation_count": len(mutations)},
                "downstream_artifacts": {
                    "target_clean_pdb": "artifacts/target_masked.pdb",
                    "target_trimmed_pdb": "artifacts/target_masked.pdb",
                    "target_masked_pdb": "artifacts/target_masked.pdb",
                    "mutations_json": "artifacts/mutations.json",
                    "mutations_csv": "artifacts/mutations.csv",
                    "target_json": "artifacts/target.json",
                },
            },
        )
    except Exception as exc:
        (job.run_dir / "stderr.log").write_text(f"{type(exc).__name__}: {exc}\n")
        finish_job(job.run_dir, False, {"outputs": {}, "metrics": {}, "error": str(exc)})
        raise
    return job.run_dir
