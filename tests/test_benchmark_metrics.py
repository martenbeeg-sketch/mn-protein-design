from __future__ import annotations

import pandas as pd
import pytest

from mn_protein_design.workflows.benchmark import (
    _benchmark_summary,
    _manual_average_precision,
    _manual_auroc,
    _metric_column_summary,
)


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [([1, 0], [0.9, 0.1], 1.0), ([1, 0], [0.1, 0.9], 0.0), ([1, 0], [0.4, 0.4], 0.5),
     ([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 1.0), ([1, 0, 1, 0], [0.8, 0.8, 0.2, 0.2], 0.5)],
)
def test_manual_auroc_matches_pairwise_ranking(labels, scores, expected: float) -> None:
    assert _manual_auroc(labels, scores) == expected


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [([1, 0], [0.9, 0.1], 1.0), ([0, 1], [0.9, 0.1], 0.5),
     ([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], (1 + 2 / 3) / 2),
     ([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5], (1 + 2 / 3) / 2)],
)
def test_manual_average_precision_ranks_positives(labels, scores, expected: float) -> None:
    assert _manual_average_precision(labels, scores) == pytest.approx(expected)


@pytest.mark.parametrize("labels,scores", [([1, 1], [0.8, 0.9]), ([0, 0], [0.8, 0.9]), ([], [])])
def test_binary_metrics_need_both_label_classes(labels, scores) -> None:
    assert _manual_auroc(labels, scores) is None
    assert _manual_average_precision(labels, scores) is None


def test_benchmark_summary_skips_unlabeled_records_and_reports_top_candidate() -> None:
    summary = _benchmark_summary([
        {"candidate_id": "weak", "label": 0, "esmfold2_benchmark_score": 0.1},
        {"candidate_id": "strong", "label": 1, "esmfold2_benchmark_score": 0.9},
        {"candidate_id": "unknown", "label": None, "esmfold2_benchmark_score": 1.0},
        {"candidate_id": "missing-score", "label": 1},
    ])
    assert summary["record_count"] == 4
    assert summary["labeled_count"] == 2
    assert summary["positive_count"] == summary["negative_count"] == 1
    assert summary["auroc"] == 1.0
    assert summary["top_candidate_id"] == "unknown"


@pytest.mark.parametrize("labels", [["yes", "no"], [True, False], [1, 0], ["binder", "nonbinder"]])
def test_metric_ranking_accepts_supported_label_spellings(labels) -> None:
    rows, summary = _metric_column_summary(pd.DataFrame({"label": labels, "metric": [1.0, 0.0]}), "label")
    assert summary["labeled_count"] == 2
    assert summary["positive_count"] == 1
    assert summary["negative_count"] == 1
    assert rows[0]["feature"] == "metric"
    assert rows[0]["best_auroc"] == 1.0


def test_metric_ranking_chooses_inverse_direction_when_lower_is_better() -> None:
    rows, summary = _metric_column_summary(pd.DataFrame({
        "label": [1, 1, 0, 0], "energy": [-5.0, -4.0, -1.0, 0.0]
    }), "label")
    assert rows[0]["feature"] == "energy"
    assert rows[0]["direction"] == "lower"
    assert summary["top_feature_direction"] == "lower"
    assert rows[0]["best_average_precision"] == 1.0


@pytest.mark.parametrize("column", ["label", "binder", "is_binder", "binds", "rf3_binder"])
def test_metric_ranking_excludes_label_leakage_columns(column: str) -> None:
    rows, summary = _metric_column_summary(
        pd.DataFrame({"label": [1, 0], column: [1, 0], "safe_metric": [0.8, 0.2]}), "label"
    )
    assert all(row["feature"] != column for row in rows)
    if column == "label":
        assert summary["top_feature"] == "safe_metric"


@pytest.mark.parametrize("column", ["rf3_chain_1_iptm", "af3_chain_2_ptm", "protenix_chain_pair_1_2_pae"])
def test_metric_ranking_excludes_chain_indexed_confidence_features(column: str) -> None:
    rows, _summary = _metric_column_summary(
        pd.DataFrame({"label": [1, 0], column: [100, 0], "global_score": [0.8, 0.2]}), "label"
    )
    assert [row["feature"] for row in rows] == ["global_score"]


@pytest.mark.parametrize(
    ("df", "label", "message"),
    [(pd.DataFrame({"score": [1, 2]}), "label", "Label column not found"),
     (pd.DataFrame({"label": [1, 1], "metric": [1, 2]}), "label", None)],
)
def test_metric_ranking_validates_label_column_and_single_class(df: pd.DataFrame, label: str, message: str | None) -> None:
    if message:
        with pytest.raises(ValueError, match=message):
            _metric_column_summary(df, label)
    else:
        rows, summary = _metric_column_summary(df, label)
        assert rows == []
        assert summary["top_feature"] is None
        assert summary["numeric_feature_count"] == 0


def test_metric_ranking_ignores_non_numeric_and_sparse_columns_and_obeys_limit() -> None:
    frame = pd.DataFrame({"label": [1, 0, 1], "text": ["a", "b", "c"], "sparse": [1, None, None],
                          "m1": [1, 0, 1], "m2": [0, 1, 0]})
    rows, summary = _metric_column_summary(frame, "label", max_columns=1)
    assert len(rows) == 1
    assert summary["numeric_feature_count"] == 2
    assert summary["scored_feature_count"] == 1


@pytest.mark.parametrize("labels,expected", [(["unknown", "unknown"], 0), ([None, "yes"], 1)])
def test_metric_ranking_counts_missing_or_unrecognized_labels(labels, expected: int) -> None:
    _rows, summary = _metric_column_summary(pd.DataFrame({"label": labels, "score": [1, 2]}), "label")
    assert summary["labeled_count"] == expected
