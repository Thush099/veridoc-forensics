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


def test_levenshtein_distance_basic_cases():
    from veridoc.benchmark import _levenshtein

    assert _levenshtein("kitten", "sitting") == 3
    assert _levenshtein("", "abc") == 3
    assert _levenshtein("same", "same") == 0
    assert _levenshtein(["a", "b"], ["a", "c", "d"]) == 2


def test_char_accuracy_is_one_for_identical_text():
    from veridoc.benchmark import compute_char_accuracy

    assert compute_char_accuracy("hello world", "hello world") == 1.0


def test_char_accuracy_normalizes_whitespace():
    from veridoc.benchmark import compute_char_accuracy

    assert compute_char_accuracy("hello   world\n", "hello world") == 1.0


def test_char_accuracy_empty_reference_contract():
    from veridoc.benchmark import compute_char_accuracy

    assert compute_char_accuracy("", "") == 1.0
    assert compute_char_accuracy("x", "") == 0.0


def test_char_accuracy_clamps_to_zero_for_wildly_wrong_text():
    from veridoc.benchmark import compute_char_accuracy

    assert compute_char_accuracy("totally different words here", "a") == 0.0


def test_word_accuracy_is_one_for_identical_words():
    from veridoc.benchmark import compute_word_accuracy

    assert compute_word_accuracy("alpha beta gamma", "alpha beta gamma") == 1.0


def test_word_accuracy_counts_substitutions():
    from veridoc.benchmark import compute_word_accuracy

    # one substitution out of three words -> 1 - 1/3
    assert abs(compute_word_accuracy("alpha x gamma", "alpha beta gamma") - (2.0 / 3.0)) < 1e-9


def test_word_accuracy_empty_reference_contract():
    from veridoc.benchmark import compute_word_accuracy

    assert compute_word_accuracy("", "") == 1.0
    assert compute_word_accuracy("word", "") == 0.0


def test_format_benchmark_table_reports_scenario_rows_and_means():
    from veridoc.benchmark import BenchmarkSuiteResult, ScenarioBenchmark, format_benchmark_table

    scenario = ScenarioBenchmark(
        scenario_id="clean_document",
        raw_char_acc=0.9,
        restored_char_acc=0.95,
        raw_word_acc=0.85,
        restored_word_acc=0.92,
        raw_ocr_conf=0.8,
        restored_ocr_conf=0.9,
        tamper_expected=False,
        tamper_detected=False,
        tamper_score=0.02,
        claim_accuracy=1.0,
        claims_matched=2,
        claims_total=2,
        faithfulness_score=1.0,
        latency_ms=123.0,
    )
    result = BenchmarkSuiteResult(
        scenarios=[scenario],
        mean_raw_char_acc=0.9,
        mean_restored_char_acc=0.95,
        mean_raw_word_acc=0.85,
        mean_restored_word_acc=0.92,
        overall_claim_accuracy=1.0,
        tamper_detection_accuracy=1.0,
        total_latency_ms=123.0,
    )
    table = format_benchmark_table(result)
    assert "clean_document" in table
    assert "CLEAN(0.02)" in table
    assert "2/2" in table
    assert "Claim Verdict Classification Acc." in table
