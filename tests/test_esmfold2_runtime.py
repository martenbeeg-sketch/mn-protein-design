from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mn_protein_design.workflows import benchmark
from mn_protein_design.workflows import esmfold2_runtime as runtime


def _write_minimal_pdb(path: Path) -> None:
    path.write_text(
        "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      1.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  C   ALA A   1      2.000   0.000   0.000  1.00 20.00           C\n"
        "TER\nEND\n"
    )


def test_stage_esmfold2_msa_decompresses_gzip_and_checks_query(tmp_path: Path) -> None:
    import gzip

    source = tmp_path / "target.a3m.gz"
    with gzip.open(source, "wt") as handle:
        handle.write(">query\nACDE\n>homolog\nAC-E\n")

    staged, note = runtime.stage_esmfold2_msa(source, "ACDE", tmp_path / "job" / "target.a3m")

    assert staged == tmp_path / "job" / "target.a3m"
    assert staged.read_text() == ">query\nACDE\n>homolog\nAC-E\n"
    assert note is None

    mismatch, note = runtime.stage_esmfold2_msa(source, "AAAA", tmp_path / "job" / "mismatch.a3m")
    assert mismatch is None
    assert note == "query_mismatch:target.a3m.gz"


def test_esmfold2_batch_uses_scheduled_gpu_and_returns_job_local_artifacts(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "job-1"
    run_dir.mkdir(parents=True)
    references = tmp_path / "references" / "biohub-esm"
    (references / "ESMFold2").mkdir(parents=True)
    (references / "ESMC-6B").mkdir()
    monkeypatch.setattr(runtime, "BIOHUB_ESM_ROOT", references)
    (run_dir / "metadata.json").write_text(json.dumps({"resource_allocation": {"gpu_device": "1"}}))

    msa_source = tmp_path / "target.a3m"
    msa_source.write_text(">query\nACDE\n>homolog\nAC-E\n")
    staged_msa, note = runtime.stage_esmfold2_msa(msa_source, "ACDE", run_dir / "inputs" / "target.a3m")
    assert note is None

    def fake_run(command, *, stdout, stderr, check):
        assert command[:3] == ["docker", "run", "--rm"]
        assert "--gpus" in command
        assert "device=1" in command
        config_arg = command[command.index("--config") + 1]
        config_path = run_dir / config_arg.removeprefix("/work/")
        config = json.loads(config_path.read_text())
        staged_container_msa = config["requests"][0]["sequences"][0]["msa_path"]
        assert staged_container_msa.startswith("/work/")
        staged_host_msa = run_dir / staged_container_msa.removeprefix("/work/")
        assert staged_host_msa.read_text() == msa_source.read_text()

        output_dir = run_dir / config["output_dir"].removeprefix("/work/")
        result_dir = output_dir / "candidate-1"
        result_dir.mkdir(parents=True)
        complex_path = result_dir / "complex.cif"
        complex_path.write_text("data_candidate-1\n")
        arrays_path = result_dir / "result_arrays.npz"
        np.savez_compressed(
            arrays_path,
            atom_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            atom_elements=np.asarray(["C"]),
            atom_names=np.asarray(["CA"]),
            token_to_atoms=np.asarray([[0, 1]], dtype=np.int32),
            chain_id=np.asarray([0], dtype=np.int32),
            plddt=np.asarray([88.0], dtype=np.float32),
            pae=np.asarray([[1.0]], dtype=np.float32),
            distogram=np.asarray([], dtype=np.float32),
            pair_chains_iptm=np.asarray([], dtype=np.float32),
        )
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "request_id": "candidate-1",
                            "complex_path": str(complex_path.relative_to(run_dir)),
                            "arrays_path": str(arrays_path.relative_to(run_dir)),
                            "complex": {
                                "id": "candidate-1",
                                "sequence": ["ACDE"],
                                "entity_lookup": {},
                                "chain_lookup": {"0": "A"},
                            },
                            "ptm": 0.7,
                            "iptm": 0.6,
                        }
                    ]
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    predictions = runtime.run_esmfold2_batch(
        run_dir=run_dir,
        requests=[
            {
                "request_id": "candidate-1",
                "sequences": [{"id": "A", "sequence": "ACDE", "msa_path": str(staged_msa)}],
            }
        ],
    )

    prediction = predictions["candidate-1"]
    assert prediction.complex_path.read_text() == "data_candidate-1\n"
    assert prediction.result.complex.chain_id.tolist() == [0]
    assert prediction.result.plddt.tolist() == [88.0]
    assert prediction.result.iptm == 0.6


def test_esmfold2_cpu_request_fails_before_docker(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "cpu-job"
    run_dir.mkdir(parents=True)
    references = tmp_path / "references" / "biohub-esm"
    (references / "ESMFold2").mkdir(parents=True)
    (references / "ESMC-6B").mkdir()
    monkeypatch.setattr(runtime, "BIOHUB_ESM_ROOT", references)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Docker should not start")))

    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        runtime.run_esmfold2_batch(
            run_dir=run_dir,
            requests=[{"request_id": "candidate-1", "sequences": [{"id": "A", "sequence": "ACDE"}]}],
            gpu_device="none",
            device="auto",
        )


def test_container_runner_loads_public_esm_pypi_api(monkeypatch, tmp_path: Path) -> None:
    from mn_protein_design.workflows import esmfold2_container_runner as runner

    references = tmp_path / "biohub-esm"
    (references / "ESMFold2").mkdir(parents=True)
    (references / "ESMC-6B").mkdir()
    calls: dict[str, object] = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["from_pretrained"] = (path, kwargs)
            return cls()

        def load_esmc(self, path, **kwargs):
            calls["load_esmc"] = (path, kwargs)

        def to(self, device):
            calls["to"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    esm = ModuleType("esm")
    esm.__path__ = []
    models = ModuleType("esm.models")
    models.__path__ = []
    esmfold2 = ModuleType("esm.models.esmfold2")
    esmfold2.EsmFold2Model = FakeModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "esm", esm)
    monkeypatch.setitem(sys.modules, "esm.models", models)
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2", esmfold2)

    loaded = runner._load_model(references, "cuda")

    assert isinstance(loaded, FakeModel)
    assert calls["from_pretrained"] == (
        str(references / "ESMFold2"),
        {"load_esmc": False, "device": "cuda"},
    )
    assert calls["load_esmc"] == (
        str(references / "ESMC-6B"),
        {"precision": "bf16"},
    )
    assert calls["to"] == "cuda"
    assert calls["eval"] is True


def test_benchmark_record_preparation_copies_msa_into_run_and_stores_relative_path(tmp_path: Path) -> None:
    import gzip

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    target = source_dir / "target.pdb"
    _write_minimal_pdb(target)
    msa = source_dir / "target.a3m.gz"
    with gzip.open(msa, "wt") as handle:
        handle.write(">query\nA\n")
    csv_path = source_dir / "benchmark.csv"
    csv_path.write_text(f"candidate_id,target_pdb,target_chain,msa_path_A\nc1,{target},A,{msa}\n")
    run_dir = tmp_path / "runs" / "job"
    input_dir = run_dir / "artifacts" / "raw" / "inputs"
    input_dir.mkdir(parents=True)

    record = benchmark._prepare_benchmark_records(csv_path, input_dir, base_dir=source_dir)[0]

    staged_msa = Path(record["msa_paths"]["A"])
    assert not staged_msa.is_absolute()
    assert (run_dir / staged_msa).is_file()
    assert (run_dir / staged_msa).read_bytes() == msa.read_bytes()


def test_esmfold2_benchmark_prepares_job_inputs_and_uses_shared_runner(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "benchmark-job"
    run_dir.mkdir()
    target_pdb = run_dir / "artifacts" / "raw" / "inputs" / "candidate-1_target.pdb"
    target_pdb.parent.mkdir(parents=True)
    _write_minimal_pdb(target_pdb)
    msa_path = run_dir / "inputs" / "target.a3m.gz"
    msa_path.parent.mkdir(parents=True)
    import gzip

    with gzip.open(msa_path, "wt") as msa_handle:
        msa_handle.write(">query\nA\n>homolog\nA\n")
    records_path = run_dir / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "candidate-1",
                    "target_pdb": str(target_pdb),
                    "target_chains": ["A"],
                    "binder_sequence": "GGG",
                    "binder_chains": ["Z"],
                    "hotspots": [],
                    "label": 1,
                    "msa_paths": {"A": str(msa_path.relative_to(run_dir))},
                }
            ]
        )
    )
    captured: dict[str, object] = {}
    complex_path = run_dir / "fake_prediction.cif"
    complex_path.write_text("data_fake\n")
    result = SimpleNamespace(complex=SimpleNamespace(), ptm=0.7, iptm=0.6, plddt=np.asarray([90.0]))

    def fake_runner(**kwargs):
        captured.update(kwargs)
        request = kwargs["requests"][0]
        assert [chain["id"] for chain in request["sequences"]] == ["A", "Z"]
        staged_msa = Path(request["sequences"][0]["msa_path"])
        assert staged_msa.is_relative_to(run_dir)
        assert staged_msa.read_text() == ">query\nA\n>homolog\nA\n"
        return {request["request_id"]: SimpleNamespace(result=result, complex_path=complex_path)}

    normalized: list[dict] = []
    monkeypatch.setattr(benchmark, "run_esmfold2_batch", fake_runner)
    monkeypatch.setattr(benchmark, "write_candidates", lambda _run, _tool, candidates: normalized.extend(candidates) or candidates)
    monkeypatch.setattr(benchmark, "finish_job", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        benchmark.refolding_workflow,
        "_esmfold2_confidence_analysis",
        lambda *_args, **_kwargs: ({}, {"mocked": True}),
    )
    monkeypatch.setattr(
        benchmark.esm_binder_workflow,
        "_hotspot_metrics_from_complex",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(benchmark.esm_binder_workflow, "_mean_plddt", lambda _result: 90.0)
    monkeypatch.setattr(benchmark, "_score_record_metrics", lambda _metrics: 0.5)

    benchmark._run_esmfold2_benchmark_worker(
        run_dir=run_dir,
        records_json=records_path,
        modes=["sequence"],
        num_loops=2,
        num_sampling_steps=16,
        seed=9,
        device="auto",
        contact_cutoff=8.0,
        use_target_msa=True,
        gpu_device="2",
    )

    assert captured["gpu_device"] == "2"
    assert captured["requests"][0]["seed"] == 9
    assert normalized[0]["target_pdb"].startswith("artifacts/raw/esmfold2_benchmark/inputs/")
    assert (run_dir / normalized[0]["target_pdb"]).is_file()
    assert (run_dir / normalized[0]["complex_pdb"]).read_text() == "data_fake\n"
    msa_artifacts = normalized[0]["raw_metadata"]["benchmark_record"]["msa_artifacts"]
    assert msa_artifacts and all(not Path(path).is_absolute() for path in msa_artifacts.values())
