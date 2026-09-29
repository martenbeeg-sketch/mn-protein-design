from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mn_protein_design.core.jobs import create_job, read_json, write_json
from mn_protein_design.core.portable_paths import is_portable_path
from mn_protein_design.workflows import bindcraft2


def _pdb_atom(serial: int, atom: str, residue: str, chain: str, number: int, x: float) -> str:
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:>4s} {residue:>3s} {chain}{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          {element:>2s}\n"
    )


def _write_target(path: Path) -> None:
    path.write_text(
        "".join(
            [
                _pdb_atom(1, "N", "ALA", "A", 1, 0.0),
                _pdb_atom(2, "CA", "ALA", "A", 1, 1.0),
                _pdb_atom(3, "C", "ALA", "A", 1, 2.0),
                _pdb_atom(4, "O", "ALA", "A", 1, 3.0),
                "TER\n",
                "END\n",
            ]
        )
    )


def test_settings_builder_uses_relative_target_and_output_paths() -> None:
    settings = bindcraft2.build_bindcraft2_settings(
        target_chains=["A", "B"],
        binder_length="55-65",
        hotspots="A12,B17",
        campaign_name="PDL1 test",
        modality="VHH",
        number_of_final_designs=3,
        max_trajectories=25,
        campaign_seed=44,
        workers_per_gpu=2,
    )

    assert settings["targets"] == [
        {
            "name": "app_target",
            "target_path": "target.pdb",
            "chains": "A,B",
            "hotspots": "A12,B17",
        }
    ]
    assert settings["binder_lengths"] == [55, 65]
    assert settings["project_folder"] == "output"
    assert settings["campaign_name"] == "PDL1 test"
    assert settings["workers_per_gpu"] == 2
    assert settings["trajectory_only"] is False


def test_settings_builder_rejects_hotspots_outside_selected_chains() -> None:
    with pytest.raises(ValueError, match="selected target chains"):
        bindcraft2.build_bindcraft2_settings(
            target_chains=["A"],
            binder_length="55",
            hotspots="B12",
        )


def test_settings_builder_supports_bc2_modality_objective_and_property_presets() -> None:
    assert set(bindcraft2.SUPPORTED_MODALITIES) == {
        "ARP", "Fab", "VHH", "binder", "cyclic_peptide", "fold_switch", "homo_oligomer",
        "induced_fit", "large_binder", "multidomain", "peptide", "scFv",
    }
    assert set(bindcraft2.BC2_PROPERTY_PRESETS) == {
        "bigbang", "disulfide_staple", "forced_targeting", "humanize", "initial_guess",
        "mixed_topology", "protease_stable", "termini_accessible", "termini_together",
    }
    settings = bindcraft2.build_bindcraft2_settings(
        target_chains=["A"],
        binder_length="65-90",
        hotspots="A12",
        modality=["binder", "induced_fit"],
        design_properties=["humanize", "protease_stable", "termini_accessible"],
    )

    assert settings["modality"] == ["binder", "induced_fit"]
    assert settings["humanize"] is True
    assert settings["protease_stable"] is True
    assert settings["termini_accessible"] is True

    native_defaults = bindcraft2.build_bindcraft2_settings(
        target_chains=["A"], binder_length=None, modality="cyclic_peptide"
    )
    assert "binder_lengths" not in native_defaults
    assert native_defaults["modality"] == "cyclic_peptide"


def test_settings_builder_validates_bc2_preset_compatibility() -> None:
    with pytest.raises(ValueError, match="requires at least one target hotspot"):
        bindcraft2.build_bindcraft2_settings(
            target_chains=["A"], binder_length="65", design_properties=["forced_targeting"]
        )
    with pytest.raises(ValueError, match="fixed scaffold modality"):
        bindcraft2.build_bindcraft2_settings(
            target_chains=["A"], binder_length="65", modality="VHH", design_properties=["mixed_topology"]
        )
    with pytest.raises(ValueError, match="currently require the binder format"):
        bindcraft2.build_bindcraft2_settings(
            target_chains=["A"], binder_length="65", modality=["VHH", "induced_fit"]
        )


def test_checkpoint_check_accepts_bc2_model_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "alphafold_models"
    model_dir.mkdir()
    for model in bindcraft2.BINDCRAFT2_MODELS:
        (model_dir / f"params_{model}.npz").write_bytes(b"x")

    assert bindcraft2.missing_bindcraft2_parameters(tmp_path, minimum_bytes=1) == []
    (model_dir / f"params_{bindcraft2.BINDCRAFT2_MODELS[0]}.npz").write_bytes(b"")
    assert bindcraft2.missing_bindcraft2_parameters(tmp_path, minimum_bytes=1) == [
        f"params_{bindcraft2.BINDCRAFT2_MODELS[0]}.npz"
    ]


def test_docker_command_mounts_reference_read_only_and_targets_selected_gpu(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference_files"
    reference_dir.mkdir()
    command = bindcraft2.build_bindcraft2_docker_command(
        run_dir=tmp_path / "runs" / "design" / "one",
        reference_dir=reference_dir,
        gpu_device="1",
    )

    assert command[:4] == ["docker", "run", "--rm", "--gpus"]
    assert command[4] == "device=1"
    assert f"{reference_dir.resolve()}:/ref:ro" in command
    assert "BINDCRAFT_AF2_PARAMS=/ref/alphafold_models" in command
    assert command[-2:] == ["design", "settings.json"]


def test_docker_command_respects_scheduler_cpu_allocation(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "runs" / "design" / "one"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "metadata.json", {"resource_allocation": {"cpu_cores": 3}})
    monkeypatch.setattr("mn_protein_design.core.scheduler._docker_cpu_quota_supported", lambda: True)

    command = bindcraft2.build_bindcraft2_docker_command(
        run_dir=run_dir,
        reference_dir=tmp_path / "references",
        gpu_device="0",
    )

    assert command[2:4] == ["--cpus", "3"]
    assert "--env" in command
    assert "OMP_NUM_THREADS=3" in command
    assert "--rm" in command
    assert "--gpus" in command


def test_enqueue_stages_target_and_portable_worker_request(
    run_store: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "source_target.pdb"
    _write_target(target)
    reference = tmp_path / "references"
    reference.mkdir()
    queued: list[Path] = []
    monkeypatch.setattr(bindcraft2, "reference_root", lambda: reference)
    monkeypatch.setattr(bindcraft2, "missing_bindcraft2_parameters", lambda root=None: [])
    monkeypatch.setattr(bindcraft2, "bindcraft2_image_available", lambda image=bindcraft2.BINDCRAFT2_IMAGE: True)
    monkeypatch.setattr(bindcraft2, "spawn_worker_for_run", queued.append)

    run_dir = bindcraft2.enqueue_bindcraft2_design(
        target_pdb=target,
        target_chains=["A"],
        binder_length="55-60",
        hotspots="A1",
        campaign_name="test design",
        modality=["binder", "induced_fit"],
        design_properties=["humanize"],
        number_of_final_designs=2,
        max_trajectories=8,
        gpu_device="1",
    )

    assert run_dir.is_relative_to(run_store)
    assert queued == [run_dir]
    staged_target = run_dir / "artifacts/raw/bindcraft2/target.pdb"
    assert staged_target.is_file()
    payload = json.loads((run_dir / "input.json").read_text())
    assert payload["inputs"]["target_pdb"] == "artifacts/raw/bindcraft2/target.pdb"
    assert payload["params"]["gpu_device"] == "1"
    assert payload["params"]["cpu_cores"] == 4
    assert payload["params"]["reference_key"] == "reference:///alphafold_models"
    settings = json.loads((run_dir / "artifacts/raw/bindcraft2/settings.json").read_text())
    assert settings["targets"][0]["target_path"] == "target.pdb"
    assert settings["project_folder"] == "output"
    assert settings["modality"] == ["binder", "induced_fit"]
    assert settings["humanize"] is True
    assert payload["params"]["design_properties"] == ["humanize"]
    assert read_json(run_dir / "worker_request.json")["kind"] == bindcraft2.BINDCRAFT2_JOB_KIND
    assert is_portable_path(read_json(run_dir / "command.json")["command"][4])


def test_local_worker_routes_bindcraft2_requests(run_store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mn_protein_design.core import local_worker

    job = create_job("design", "bindcraft2_design", "bindcraft2", {}, {"gpu_device": "0"})
    write_json(job.run_dir / "worker_request.json", {"kind": bindcraft2.BINDCRAFT2_JOB_KIND})
    called: list[Path] = []
    monkeypatch.setattr(bindcraft2, "run_bindcraft2_job", called.append)

    assert local_worker.run_worker_job(job.run_dir) == 0
    assert called == [job.run_dir]


def test_ranked_mmcif_rows_become_normalized_candidates(
    run_store: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Bio.PDB import MMCIFIO, PDBParser
    from mn_protein_design.core import portable_paths

    monkeypatch.setattr(portable_paths, "runs_root", lambda: run_store)

    run_dir = run_store / "design" / "bc2-test"
    raw = run_dir / "artifacts/raw/bindcraft2"
    ranked = raw / "output/3_Ranked"
    ranked.mkdir(parents=True)
    target = raw / "target.pdb"
    _write_target(target)
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "".join(
            [
                _pdb_atom(1, "N", "ALA", "A", 1, 0.0),
                _pdb_atom(2, "CA", "ALA", "A", 1, 1.0),
                _pdb_atom(3, "C", "ALA", "A", 1, 2.0),
                _pdb_atom(4, "O", "ALA", "A", 1, 3.0),
                _pdb_atom(5, "N", "GLY", "B", 1, 5.0),
                _pdb_atom(6, "CA", "GLY", "B", 1, 6.0),
                _pdb_atom(7, "C", "GLY", "B", 1, 7.0),
                _pdb_atom(8, "O", "GLY", "B", 1, 8.0),
                "TER\n",
                "END\n",
            ]
        )
    )
    structure = PDBParser(QUIET=True).get_structure("complex", str(complex_pdb))
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(ranked / "design_seq1.cif"))
    with (ranked / "!_Ranked.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "design", "length", "Binder_Sequence", "i_pDAE"])
        writer.writeheader()
        writer.writerow({"rank": "1", "design": "design_seq1", "length": "1", "Binder_Sequence": "G", "i_pDAE": "0.75"})

    candidates = bindcraft2.normalize_bindcraft2_candidates(
        run_dir,
        {"target_chains": ["A"], "binder_length": "1", "hotspots": "A1"},
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["stage"] == "complex_refolding"
    assert candidate["source_tool"] == "bindcraft2"
    assert candidate["binder_sequence"] == "G"
    assert candidate["target_chains"] == ["A"]
    assert candidate["binder_chains"] == ["B"]
    assert candidate["metrics"]["bindcraft2_rank"] == 1
    assert candidate["metrics"]["i_pDAE"] == 0.75
    assert is_portable_path(candidate["complex_pdb"])
