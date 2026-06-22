from __future__ import annotations

from io import StringIO
import math
from pathlib import Path
import urllib.request

from mn_protein_design.core.residue_selection import ResidueSelection


WATER_NAMES = {"HOH", "WAT", "H2O"}
STANDARD_PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
MODIFIED_RESIDUE_MAPPINGS = {
    "CAS": {
        "target": "CYS",
        "keep_atoms": {"N", "CA", "C", "O", "CB", "SG"},
        "description": "CAS mapped to CYS by keeping protein-compatible atoms and dropping arsenic substituent atoms.",
    },
}


def pdb_chains(path: Path) -> list[str]:
    chains: set[str] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
            chain = line[21].strip()
            if chain:
                chains.add(chain)
    return sorted(chains)


def download_pdb(pdb_id: str) -> str:
    code = pdb_id.strip().upper()
    if not code or len(code) != 4 or not code.isalnum():
        raise ValueError("PDB ID must be a four-character code, for example 1BRS.")
    url = f"https://files.rcsb.org/download/{code}.pdb"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read().decode("utf-8")
    if not data.startswith(("HEADER", "ATOM", "TITLE", "REMARK")):
        raise ValueError(f"Downloaded response for {code} does not look like a PDB file.")
    return data


def download_alphafold_db_pdb(uniprot_id_or_url: str) -> str:
    text = str(uniprot_id_or_url or "").strip()
    if not text:
        raise ValueError("Provide a UniProt accession or an AlphaFold DB PDB URL.")
    if text.startswith(("http://", "https://")):
        url = text
        label = Path(url.rstrip("/")).name or "AlphaFold DB structure"
    else:
        accession = text.upper()
        if not accession.replace("-", "").isalnum():
            raise ValueError("UniProt accession should contain only letters, numbers, or hyphens.")
        url = f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb"
        label = accession
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read().decode("utf-8")
    if "ATOM" not in data[:20000]:
        raise ValueError(f"Downloaded response for {label} does not look like a PDB file.")
    return data


def pdb_summary(pdb_text: str) -> dict:
    residues_by_chain: dict[str, set[int]] = {}
    atom_count = 0
    hetero_count = 0
    water_count = 0
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        atom_count += record == "ATOM"
        hetero_count += record == "HETATM"
        chain = line[21].strip() if len(line) > 21 else ""
        resname = line[17:20].strip() if len(line) >= 20 else ""
        if resname in WATER_NAMES:
            water_count += 1
        try:
            resseq = int(line[22:26].strip())
        except ValueError:
            continue
        if chain:
            residues_by_chain.setdefault(chain, set()).add(resseq)
    chains = []
    for chain, residues in sorted(residues_by_chain.items()):
        sorted_residues = sorted(residues)
        chains.append(
            {
                "chain_id": chain,
                "residue_count": len(sorted_residues),
                "start": sorted_residues[0] if sorted_residues else None,
                "end": sorted_residues[-1] if sorted_residues else None,
                "residues": sorted_residues,
            }
        )
    return {"chains": chains, "atom_count": atom_count, "hetero_count": hetero_count, "water_count": water_count}


def detect_nonstandard_residues(pdb_text: str) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str], set[str]] = {}
    first_record: dict[tuple[str, str, str, str, str], str] = {}
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        resname = line[17:20].strip()
        if resname in STANDARD_PROTEIN_RESIDUES or resname in WATER_NAMES:
            continue
        chain = line[21].strip() if len(line) > 21 else ""
        resseq = line[22:26].strip() if len(line) >= 26 else ""
        icode = line[26].strip() if len(line) > 26 else ""
        atom_name = line[12:16].strip()
        key = (chain, resseq, icode, resname, record)
        grouped.setdefault(key, set()).add(atom_name)
        first_record.setdefault(key, record)

    residues: list[dict] = []
    for (chain, resseq, icode, resname, record), atoms in grouped.items():
        is_protein_like = bool({"N", "CA", "C"}.issubset(atoms) or record == "ATOM")
        if not is_protein_like:
            continue
        mapping = MODIFIED_RESIDUE_MAPPINGS.get(resname)
        residues.append(
            {
                "record": record,
                "chain": chain,
                "residue_number": resseq,
                "insertion_code": icode,
                "resname": resname,
                "known_mapping": bool(mapping),
                "mapped_to": mapping["target"] if mapping else "",
            }
        )
    return residues


def known_modified_residue_names() -> set[str]:
    return set(MODIFIED_RESIDUE_MAPPINGS)


def _pdb_atom_occupancy(line: str) -> tuple[int, float]:
    altloc = line[16].strip() if len(line) > 16 else ""
    if altloc == "":
        altloc_rank = 3
    elif altloc == "A":
        altloc_rank = 2
    else:
        altloc_rank = 1
    try:
        occupancy = float(line[54:60].strip())
    except ValueError:
        occupancy = 0.0
    return altloc_rank, occupancy


def map_modified_residues_to_standard(pdb_text: str) -> tuple[str, dict]:
    """Map known non-canonical amino acids to conservative canonical forms.

    The mappings are residue-specific and intentionally narrow. For CAS, this
    mirrors mn-ligand: keep the CYS-compatible backbone/sidechain atoms and drop
    the arsenic substituent atoms.
    """
    selected_atoms: dict[tuple[str, str, str, str, str], tuple[tuple[int, float], int, str]] = {}
    mapped_residues: dict[tuple[str, str, str, str], dict] = {}

    for idx, line in enumerate(pdb_text.splitlines()):
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"} or len(line) < 27:
            continue
        resname = line[17:20].strip()
        mapping = MODIFIED_RESIDUE_MAPPINGS.get(resname)
        if not mapping:
            continue
        chain = line[21].strip()
        resseq = line[22:26].strip()
        icode = line[26].strip()
        atom_name = line[12:16].strip()
        residue_key = (resname, chain, resseq, icode)
        report = mapped_residues.setdefault(
            residue_key,
            {
                "resname": resname,
                "target": mapping["target"],
                "chain": chain,
                "residue_number": resseq,
                "insertion_code": icode,
                "kept_atoms": [],
                "dropped_atoms": 0,
            },
        )
        if atom_name not in mapping["keep_atoms"]:
            report["dropped_atoms"] += 1
            continue
        group_key = (*residue_key, atom_name)
        converted = "ATOM  " + line[6:16] + " " + f"{mapping['target']:>3}" + line[20:]
        ranked = (_pdb_atom_occupancy(line), idx, converted)
        if group_key not in selected_atoms or ranked[0] > selected_atoms[group_key][0]:
            selected_atoms[group_key] = ranked

    selected_by_index = {idx: line for _, idx, line in selected_atoms.values()}
    emitted_groups: set[tuple[str, str, str, str, str]] = set()
    output_lines: list[str] = []
    for idx, line in enumerate(pdb_text.splitlines()):
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM", "ANISOU"} or len(line) < 27:
            output_lines.append(line)
            continue
        resname = line[17:20].strip()
        mapping = MODIFIED_RESIDUE_MAPPINGS.get(resname)
        if not mapping:
            output_lines.append(line)
            continue
        atom_name = line[12:16].strip()
        residue_key = (resname, line[21].strip(), line[22:26].strip(), line[26].strip())
        if atom_name not in mapping["keep_atoms"]:
            continue
        if record == "ANISOU":
            continue
        group_key = (*residue_key, atom_name)
        if idx not in selected_by_index:
            continue
        if group_key in emitted_groups:
            continue
        emitted_groups.add(group_key)
        mapped_residues[residue_key]["kept_atoms"].append(atom_name)
        output_lines.append(selected_by_index[idx])

    residue_reports = list(mapped_residues.values())
    for report in residue_reports:
        report["kept_atoms"] = sorted(set(report["kept_atoms"]))
    return "\n".join(output_lines) + ("\n" if output_lines else ""), {
        "modified_residue_mapping_enabled": True,
        "modified_residue_mapping_count": len(residue_reports),
        "modified_residue_mappings": residue_reports,
    }


def replace_nonstandard_residues_with_pdbfixer(pdb_text: str) -> tuple[str, dict]:
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise ImportError("PDBFixer/OpenMM is not installed. Update the mn-protein-design Conda environment.") from exc

    fixer = PDBFixer(pdbfile=StringIO(pdb_text))
    fixer.findNonstandardResidues()
    replacements = [
        {
            "chain": residue.chain.id,
            "residue_number": str(residue.id),
            "resname": residue.name,
            "replacement": replacement,
        }
        for residue, replacement in fixer.nonstandardResidues
    ]
    fixer.replaceNonstandardResidues()
    output = StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, output, keepIds=True)
    return output.getvalue(), {"pdbfixer_nonstandard_replacements": replacements, "pdbfixer_replacement_count": len(replacements)}


def filter_pdb_text(
    pdb_text: str,
    keep_chains: set[str] | None = None,
    remove_waters: bool = True,
    remove_hetero: bool = False,
    residue_selections: list[ResidueSelection] | None = None,
) -> str:
    residue_selections = residue_selections or []
    output: list[str] = []
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record in {"ATOM", "HETATM", "TER", "ANISOU"}:
            chain = line[21].strip() if len(line) > 21 else ""
            resname = line[17:20].strip() if len(line) >= 20 else ""
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                resseq = None
            if keep_chains and chain not in keep_chains:
                continue
            if remove_waters and resname in WATER_NAMES:
                continue
            if remove_hetero and record == "HETATM":
                continue
            if residue_selections and resseq is not None and not any(s.contains(chain, resseq) for s in residue_selections):
                continue
            output.append(line)
        elif record in {"HEADER", "TITLE", "REMARK", "CRYST1", "MODEL", "ENDMDL", "END"}:
            output.append(line)
    if output and output[-1][:6].strip() != "END":
        output.append("END")
    return "\n".join(output) + ("\n" if output else "")


def residue_inventory(pdb_text: str) -> list[dict]:
    residues: dict[tuple[str, int, str, str], dict] = {}
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        try:
            resseq = int(line[22:26].strip())
        except ValueError:
            continue
        chain = line[21].strip() if len(line) > 21 else ""
        icode = line[26].strip() if len(line) > 26 else ""
        resname = line[17:20].strip() if len(line) >= 20 else ""
        key = (chain, resseq, icode, resname)
        residues.setdefault(
            key,
            {"chain": chain, "residue_number": resseq, "insertion_code": icode, "resname": resname, "atom_count": 0},
        )["atom_count"] += 1
    return sorted(residues.values(), key=lambda row: (row["chain"], row["residue_number"], row["insertion_code"], row["resname"]))


def _pdb_residue_anchor_points(pdb_text: str) -> dict[tuple[str, int], tuple[float, float, float]]:
    residue_atoms: dict[tuple[str, int], list[tuple[str, tuple[float, float, float]]]] = {}
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        resname = line[17:20].strip() if len(line) >= 20 else ""
        if resname in WATER_NAMES:
            continue
        try:
            resseq = int(line[22:26].strip())
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        chain = line[21].strip() if len(line) > 21 else ""
        atom_name = line[12:16].strip()
        residue_atoms.setdefault((chain, resseq), []).append((atom_name, xyz))

    anchors: dict[tuple[str, int], tuple[float, float, float]] = {}
    for residue_key, atoms in residue_atoms.items():
        ca_atoms = [xyz for atom_name, xyz in atoms if atom_name == "CA"]
        if ca_atoms:
            anchors[residue_key] = ca_atoms[0]
            continue
        anchors[residue_key] = (
            sum(xyz[0] for _atom_name, xyz in atoms) / len(atoms),
            sum(xyz[1] for _atom_name, xyz in atoms) / len(atoms),
            sum(xyz[2] for _atom_name, xyz in atoms) / len(atoms),
        )
    return anchors


def residues_within_spheres(pdb_text: str, seeds: set[tuple[str, int]], diameter_angstrom: float) -> set[tuple[str, int]]:
    if diameter_angstrom <= 0 or not seeds:
        return set()
    radius = diameter_angstrom / 2.0
    radius_sq = radius * radius
    anchors_by_residue = _pdb_residue_anchor_points(pdb_text)
    seed_anchors = [anchors_by_residue[seed] for seed in seeds if seed in anchors_by_residue]
    if not seed_anchors:
        return set()
    selected: set[tuple[str, int]] = set()
    for residue_key, anchor in anchors_by_residue.items():
        for seed_anchor in seed_anchors:
            dist_sq = (
                math.pow(anchor[0] - seed_anchor[0], 2)
                + math.pow(anchor[1] - seed_anchor[1], 2)
                + math.pow(anchor[2] - seed_anchor[2], 2)
            )
            if dist_sq <= radius_sq:
                selected.add(residue_key)
                break
    return selected


def residues_on_plane_sides(
    pdb_text: str,
    plane_residues: set[tuple[str, int]],
    tolerance_angstrom: float = 0.5,
) -> tuple[dict[str, set[tuple[str, int]]], dict]:
    """Split residue anchors by the plane defined by three selected residues."""
    if len(plane_residues) != 3:
        raise ValueError("Select exactly three residues to define a plane.")

    anchors_by_residue = _pdb_residue_anchor_points(pdb_text)
    missing = sorted(residue for residue in plane_residues if residue not in anchors_by_residue)
    if missing:
        labels = ", ".join(f"{chain}{residue}" for chain, residue in missing)
        raise ValueError(f"Plane residue anchors are missing: {labels}")

    ordered_plane_residues = sorted(plane_residues)
    p1, p2, p3 = [anchors_by_residue[residue] for residue in ordered_plane_residues]

    v1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    v2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
    normal = (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )
    normal_length = math.sqrt(
        normal[0] * normal[0]
        + normal[1] * normal[1]
        + normal[2] * normal[2]
    )
    if normal_length < 1e-6:
        raise ValueError("The selected plane residues are collinear; choose a wider triangle.")

    normal_unit = (
        normal[0] / normal_length,
        normal[1] / normal_length,
        normal[2] / normal_length,
    )

    side_a: set[tuple[str, int]] = set()
    side_b: set[tuple[str, int]] = set()
    on_plane: set[tuple[str, int]] = set()
    tolerance = max(0.0, tolerance_angstrom)

    for residue_key, anchor in anchors_by_residue.items():
        delta = (anchor[0] - p1[0], anchor[1] - p1[1], anchor[2] - p1[2])
        signed_distance = (
            delta[0] * normal_unit[0]
            + delta[1] * normal_unit[1]
            + delta[2] * normal_unit[2]
        )

        if signed_distance >= -tolerance:
            side_a.add(residue_key)
        if signed_distance <= tolerance:
            side_b.add(residue_key)
        if abs(signed_distance) <= tolerance:
            on_plane.add(residue_key)

    plane_info = {
        "plane_residues": [
            {"chain": chain, "residue_number": residue}
            for chain, residue in ordered_plane_residues
        ],
        "normal": [round(value, 6) for value in normal_unit],
        "tolerance_angstrom": tolerance,
        "on_plane_count": len(on_plane),
    }

    return {"side_a": side_a, "side_b": side_b, "on_plane": on_plane}, plane_info


def filter_pdb_to_residues(
    pdb_text: str,
    selected_residues: set[tuple[str, int]],
    remove_waters: bool = True,
    remove_hetero: bool = False,
) -> tuple[str, dict]:
    output: list[str] = []
    stats = {"input_lines": 0, "written_lines": 0, "removed_lines": 0, "chains": set(), "residues": set()}
    for line in pdb_text.splitlines():
        stats["input_lines"] += 1
        record = line[:6].strip()
        if record in {"ATOM", "HETATM", "ANISOU", "TER"}:
            chain = line[21].strip() if len(line) > 21 else ""
            resname = line[17:20].strip() if len(line) >= 20 else ""
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                resseq = None
            if remove_waters and resname in WATER_NAMES:
                stats["removed_lines"] += 1
                continue
            if remove_hetero and record == "HETATM":
                stats["removed_lines"] += 1
                continue
            if resseq is None or (chain, resseq) not in selected_residues:
                stats["removed_lines"] += 1
                continue
            output.append(line)
            stats["written_lines"] += 1
            if record in {"ATOM", "HETATM"}:
                stats["chains"].add(chain)
                stats["residues"].add((chain, resseq))
        elif record in {"HEADER", "TITLE", "REMARK", "CRYST1", "MODEL", "ENDMDL", "END"}:
            output.append(line)
            stats["written_lines"] += 1
        else:
            stats["removed_lines"] += 1
    if output and output[-1][:6].strip() != "END":
        output.append("END")
    return "\n".join(output) + ("\n" if output else ""), {
        "input_lines": stats["input_lines"],
        "written_lines": stats["written_lines"],
        "removed_lines": stats["removed_lines"],
        "chains": sorted(c for c in stats["chains"] if c),
        "residue_count": len(stats["residues"]),
    }


def sanitize_pdb_for_surface_tools(pdb_text: str) -> tuple[str, dict]:
    """Return a PDB text suitable for APBS/MSMS/Surf2Spot-style surface tools.

    Surface pipelines tend to be brittle around alternate locations and ANISOU
    records. Keep one altloc per atom site, preferring blank altlocs, then A,
    then highest occupancy, and blank the altloc column in the output.
    """
    passthrough: list[tuple[int, str]] = []
    atom_records: dict[tuple[str, str, str, str, str, str], tuple[tuple[int, int, float], int, str]] = {}
    altloc_atoms = 0
    anisou_records = 0
    atom_lines = 0

    def score(line: str) -> tuple[int, int, float]:
        altloc = line[16].strip() if len(line) > 16 else ""
        try:
            occupancy = float(line[54:60].strip())
        except ValueError:
            occupancy = 0.0
        if altloc == "":
            alt_rank = 3
        elif altloc == "A":
            alt_rank = 2
        else:
            alt_rank = 1
        return alt_rank, 1 if occupancy >= 0 else 0, occupancy

    lines = pdb_text.splitlines()
    for idx, line in enumerate(lines):
        record = line[:6].strip()
        if record == "ANISOU":
            anisou_records += 1
            continue
        if record in {"ATOM", "HETATM"}:
            atom_lines += 1
            if len(line) > 16 and line[16].strip():
                altloc_atoms += 1
            key = (
                record,
                line[21:22],
                line[22:26],
                line[26:27],
                line[17:20],
                line[12:16],
            )
            ranked = (score(line), idx, line[:16] + " " + line[17:] if len(line) > 16 else line)
            if key not in atom_records or ranked[0] > atom_records[key][0]:
                atom_records[key] = ranked
            continue
        if record in {"HEADER", "TITLE", "REMARK", "CRYST1", "MODEL", "ENDMDL", "END", "TER"}:
            passthrough.append((idx, line))

    selected_atoms = [(idx, line) for _, idx, line in atom_records.values()]
    output_pairs = sorted(passthrough + selected_atoms, key=lambda item: item[0])
    output = [line for _, line in output_pairs]
    if output and output[-1][:6].strip() != "END":
        output.append("END")
    metrics = {
        "input_atom_lines": atom_lines,
        "output_atom_lines": len(selected_atoms),
        "removed_altloc_atom_lines": atom_lines - len(selected_atoms),
        "input_altloc_atom_lines": altloc_atoms,
        "removed_anisou_lines": anisou_records,
    }
    return "\n".join(output) + ("\n" if output else ""), metrics


def clean_pdb(
    input_pdb: Path,
    output_pdb: Path,
    keep_chains: set[str] | None = None,
    remove_waters: bool = True,
    remove_hetero: bool = False,
    residue_selections: list[ResidueSelection] | None = None,
) -> dict:
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    residue_selections = residue_selections or []
    stats = {"input_lines": 0, "written_lines": 0, "removed_lines": 0, "chains": set(), "residues": set()}
    with input_pdb.open("r", errors="ignore") as source, output_pdb.open("w") as dest:
        for line in source:
            stats["input_lines"] += 1
            record = line[:6].strip()
            if record in {"ATOM", "HETATM", "TER", "ANISOU"}:
                chain = line[21].strip() if len(line) > 21 else ""
                resname = line[17:20].strip() if len(line) >= 20 else ""
                resseq_text = line[22:26].strip() if len(line) >= 26 else ""
                try:
                    resseq = int(resseq_text)
                except ValueError:
                    resseq = None
                if keep_chains and chain not in keep_chains:
                    stats["removed_lines"] += 1
                    continue
                if remove_waters and resname in WATER_NAMES:
                    stats["removed_lines"] += 1
                    continue
                if remove_hetero and record == "HETATM":
                    stats["removed_lines"] += 1
                    continue
                if residue_selections and resseq is not None and not any(s.contains(chain, resseq) for s in residue_selections):
                    stats["removed_lines"] += 1
                    continue
                if record in {"ATOM", "HETATM"}:
                    stats["chains"].add(chain)
                    if resseq is not None:
                        stats["residues"].add((chain, resseq))
                dest.write(line)
                stats["written_lines"] += 1
            elif record in {"HEADER", "TITLE", "REMARK", "CRYST1", "MODEL", "ENDMDL", "END"}:
                dest.write(line)
                stats["written_lines"] += 1
            else:
                stats["removed_lines"] += 1
    return {
        "input_lines": stats["input_lines"],
        "written_lines": stats["written_lines"],
        "removed_lines": stats["removed_lines"],
        "chains": sorted(c for c in stats["chains"] if c),
        "residue_count": len(stats["residues"]),
    }
