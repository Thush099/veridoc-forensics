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


def _bench(claim_accuracy: float):
    from veridoc.benchmark import BenchmarkSuiteResult

    return BenchmarkSuiteResult(
        scenarios=[],
        mean_raw_char_acc=0.0,
        mean_restored_char_acc=0.0,
        mean_raw_word_acc=0.0,
        mean_restored_word_acc=0.0,
        overall_claim_accuracy=claim_accuracy,
        tamper_detection_accuracy=0.0,
        total_latency_ms=0.0,
    )


def test_exit_code_no_threshold_always_succeeds():
    from main import exit_code_for

    assert exit_code_for(_bench(0.0), None) == 0


def test_exit_code_gates_on_min_claim_accuracy():
    from main import exit_code_for

    assert exit_code_for(_bench(0.5), 0.95) == 1
    assert exit_code_for(_bench(0.95), 0.95) == 0
    assert exit_code_for(_bench(1.0), 0.95) == 0


def test_exit_code_rejects_invalid_threshold():
    import pytest

    from main import exit_code_for

    for bad in (float("nan"), float("inf"), float("-inf"), -0.1, 1.5):
        with pytest.raises(ValueError, match="min_claim_accuracy must be a finite value"):
            exit_code_for(_bench(1.0), bad)
