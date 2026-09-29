from __future__ import annotations

import json
from pathlib import Path

import pytest

from mn_protein_design.core.jobs import create_job, read_json
from mn_protein_design.workflows import bindcraft2, design_campaigns


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


def test_bindcraft2_is_in_vanilla_registry_but_excluded_from_generator_only() -> None:
    assert "bindcraft2" in design_campaigns.ENGINE_ORDER
    assert design_campaigns.ENGINE_LABELS["bindcraft2"] == "BindCraft 2"
    assert "bindcraft2" in design_campaigns.VANILLA_ONLY_ENGINES
    assert "bindcraft2" not in design_campaigns.LEVEL1_SEQUENCE_REQUIRED_ENGINES


def test_create_campaign_records_bindcraft2_cpu_request(
    run_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queued: list[Path] = []
    monkeypatch.setattr(design_campaigns, "spawn_worker_for_run", queued.append)

    run_dir = design_campaigns.create_design_campaign(
        target_pdb=run_store / "target.pdb",
        target_chains=["A"],
        binder_length="55-65",
        hotspots="A12",
        campaign_name="BC2 campaign",
        design_attempts=12,
        sequences_per_backbone=4,
        random_seed=19,
        engines=["bindcraft2"],
        engine_configs={
            "bindcraft2": {
                "modality": ["binder", "fold_switch"],
                "design_properties": ["humanize"],
                "number_of_final_designs": 3,
                "max_trajectories": 12,
                "campaign_seed": 19,
                "workers_per_gpu": 2,
                "cpu_cores": 3,
            }
        },
    )

    params = read_json(run_dir / "input.json")["params"]
    assert queued == [run_dir]
    assert params["engines"] == ["bindcraft2"]
    assert params["cpu_cores"] == 3
    assert params["engine_configs"]["bindcraft2"]["modality"] == ["binder", "fold_switch"]
    assert params["engine_configs"]["bindcraft2"]["design_properties"] == ["humanize"]


def test_create_campaign_rejects_bindcraft2_from_staged_recipe(run_store: Path) -> None:
    with pytest.raises(ValueError, match="only in the vanilla campaign workflow"):
        design_campaigns.create_design_campaign(
            target_pdb=run_store / "target.pdb",
            target_chains=["A"],
            binder_length="55-65",
            hotspots="",
            campaign_name="BC2 staged",
            design_attempts=2,
            sequences_per_backbone=1,
            random_seed=1,
            engines=["bindcraft2"],
            engine_configs={},
            workflow_recipe="staged",
        )


def test_campaign_engine_dispatch_runs_bindcraft2_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_run = tmp_path / "campaign"
    parent_run.mkdir()
    captured: dict = {}

    def run_inline(**kwargs):
        captured.update(kwargs)
        return tmp_path / "bindcraft2-child"

    monkeypatch.setattr(bindcraft2, "run_bindcraft2_campaign", run_inline, raising=False)
    target = tmp_path / "target.pdb"
    _write_target(target)

    result = design_campaigns._run_engine(
        "bindcraft2",
        target_pdb=target,
        target_chains=["A"],
        binder_length="55-65",
        hotspots="A12",
        campaign_name="campaign",
        design_attempts=9,
        sequences_per_backbone=2,
        random_seed=20,
        gpu_device="1",
        config={"modality": ["binder", "induced_fit"], "design_properties": ["humanize"], "workers_per_gpu": 2, "cpu_cores": 3},
        parent_run_dir=parent_run,
    )

    assert result == tmp_path / "bindcraft2-child"
    assert captured["parent_run_dir"] == parent_run
    assert captured["target_chains"] == ["A"]
    assert captured["hotspots"] == "A12"
    assert captured["modality"] == ["binder", "induced_fit"]
    assert captured["design_properties"] == ["humanize"]
    assert captured["max_trajectories"] == 9
    assert captured["workers_per_gpu"] == 2
    assert captured["cpu_cores"] == 3
    assert captured["gpu_device"] == "1"


def test_bindcraft2_campaign_child_runs_inline_and_inherits_allocation(
    run_store: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.pdb"
    _write_target(target)
    references = tmp_path / "references"
    references.mkdir()
    monkeypatch.setattr(bindcraft2, "reference_root", lambda: references)
    monkeypatch.setattr(bindcraft2, "missing_bindcraft2_parameters", lambda root=None: [])
    monkeypatch.setattr(bindcraft2, "bindcraft2_image_available", lambda image=bindcraft2.BINDCRAFT2_IMAGE: True)
    monkeypatch.setattr(bindcraft2, "spawn_worker_for_run", lambda _path: pytest.fail("inline campaign must not spawn a second worker"))

    parent = create_job(
        "design-campaign",
        "multi_engine_design_campaign",
        "design_campaign",
        {},
        {"gpu_device": "1", "cpu_cores": 3},
    )
    parent_metadata = read_json(parent.run_dir / "metadata.json")
    parent_metadata["resource_allocation"] = {
        "cpu_cores": 3,
        "gpu_device": "1",
        "gpu_resource": "gpu:1",
        "scheduler": "local-worker-service",
    }
    (parent.run_dir / "metadata.json").write_text(json.dumps(parent_metadata))
    executed: list[Path] = []
    monkeypatch.setattr(bindcraft2, "run_bindcraft2_job", executed.append)

    child = bindcraft2.run_bindcraft2_campaign(
        parent_run_dir=parent.run_dir,
        target_pdb=target,
        target_chains=["A"],
        binder_length="55-60",
        campaign_name="BC2 campaign",
        modality="binder",
        number_of_final_designs=2,
        max_trajectories=8,
        campaign_seed=5,
        workers_per_gpu=2,
        cpu_cores=3,
        gpu_device="1",
    )

    assert executed == [child]
    assert not (child / "worker_request.json").exists()
    assert read_json(child / "input.json")["params"]["gpu_device"] == "1"
    assert read_json(child / "metadata.json")["resource_allocation"]["cpu_cores"] == 3
    assert read_json(child / "metadata.json")["resource_allocation"]["gpu_device"] == "1"
