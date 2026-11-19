"""Tests for benchmark claim-verdict scoring."""

from veridoc.benchmark import _score_verdicts


def test_score_verdicts_counts_unmatched_extra_predictions_as_incorrect():
    matched, total = _score_verdicts(["SUPPORTED"], ["SUPPORTED", "UNVERIFIED"])

    assert matched == 1
    assert total == 2


def test_score_verdicts_counts_missing_predictions_as_incorrect():
    matched, total = _score_verdicts(["SUPPORTED", "CONTRADICTED"], ["SUPPORTED"])

    assert matched == 1
    assert total == 2
