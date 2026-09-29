from __future__ import annotations

import hashlib
import io

import pytest

from mn_protein_design.workflows import benchmark


def test_published_dataset_is_downloaded_into_managed_path(monkeypatch, tmp_path) -> None:
    payload = b"binder,label\nAAA,1\n"
    target = tmp_path / "reference_files" / "benchmark" / "dataset.csv"
    monkeypatch.setattr(benchmark, "PUBLISHED_DATASET", target)
    monkeypatch.setattr(benchmark, "PUBLISHED_DATASET_URL", "https://example.invalid/dataset.csv")
    monkeypatch.setattr(benchmark, "PUBLISHED_DATASET_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(benchmark, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload))

    result = benchmark._ensure_published_benchmark_dataset()

    assert result == target
    assert target.read_bytes() == payload
    assert not target.with_suffix(".csv.download").exists()


def test_published_dataset_checksum_mismatch_leaves_no_partial_file(monkeypatch, tmp_path) -> None:
    payload = b"untrusted content"
    target = tmp_path / "dataset.csv"
    monkeypatch.setattr(benchmark, "PUBLISHED_DATASET", target)
    monkeypatch.setattr(benchmark, "PUBLISHED_DATASET_URL", "https://example.invalid/dataset.csv")
    monkeypatch.setattr(benchmark, "PUBLISHED_DATASET_SHA256", "0" * 64)
    monkeypatch.setattr(benchmark, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload))

    with pytest.raises(ValueError, match="checksum"):
        benchmark._ensure_published_benchmark_dataset()

    assert not target.exists()
    assert not target.with_suffix(".csv.download").exists()
