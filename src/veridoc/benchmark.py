"""Reproducible benchmark measuring raw vs forensic-restored OCR and claim verification accuracy."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass

from veridoc.forensics import DocumentForensicAnalyzer
from veridoc.ocr import OCRBackend, TesseractExtractor
from veridoc.pipeline import VeriDocPipeline
from veridoc.synthetic import generate_demo_suite
from veridoc.verifier import split_claims


def _levenshtein(seq_a: list[str] | str, seq_b: list[str] | str) -> int:
    """Standard dynamic-programming Levenshtein distance in O(min(m, n)) space."""
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for item_a in seq_a:
        curr = [prev[0] + 1]
        for j, item_b in enumerate(seq_b):
            cost = 0 if item_a == item_b else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_char_accuracy(hypothesis: str, reference: str) -> float:
    """Compute character accuracy ``max(0.0, 1.0 - CER)`` over whitespace-normalized text."""
    hyp = re.sub(r"\s+", " ", hypothesis).strip()
    ref = re.sub(r"\s+", " ", reference).strip()
    if not ref:
        return 1.0 if not hyp else 0.0
    dist = _levenshtein(hyp, ref)
    return max(0.0, 1.0 - dist / len(ref))


def compute_word_accuracy(hypothesis: str, reference: str) -> float:
    """Compute word accuracy ``max(0.0, 1.0 - WER)`` over whitespace-tokenized words."""
    hyp_words = re.sub(r"\s+", " ", hypothesis).strip().split()
    ref_words = re.sub(r"\s+", " ", reference).strip().split()
    if not ref_words:
        return 1.0 if not hyp_words else 0.0
    dist = _levenshtein(hyp_words, ref_words)
    return max(0.0, 1.0 - dist / len(ref_words))


@dataclass
class ScenarioBenchmark:
    """Measured telemetry for one synthetic document scenario."""

    scenario_id: str
    raw_char_acc: float
    restored_char_acc: float
    raw_word_acc: float
    restored_word_acc: float
    raw_ocr_conf: float
    restored_ocr_conf: float
    tamper_expected: bool
    tamper_detected: bool
    tamper_score: float
    claim_accuracy: float
    claims_matched: int
    claims_total: int
    faithfulness_score: float
    latency_ms: float


@dataclass
class BenchmarkSuiteResult:
    """Aggregate benchmark summary across all scenarios."""

    scenarios: list[ScenarioBenchmark]
    mean_raw_char_acc: float
    mean_restored_char_acc: float
    mean_raw_word_acc: float
    mean_restored_word_acc: float
    overall_claim_accuracy: float
    tamper_detection_accuracy: float
    total_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "scenarios": [asdict(s) for s in self.scenarios],
            "mean_raw_char_acc": self.mean_raw_char_acc,
            "mean_restored_char_acc": self.mean_restored_char_acc,
            "mean_raw_word_acc": self.mean_raw_word_acc,
            "mean_restored_word_acc": self.mean_restored_word_acc,
            "overall_claim_accuracy": self.overall_claim_accuracy,
            "tamper_detection_accuracy": self.tamper_detection_accuracy,
            "total_latency_ms": self.total_latency_ms,
        }


def run_benchmark(
    analyzer: DocumentForensicAnalyzer | None = None,
    ocr_backend: OCRBackend | None = None,
) -> BenchmarkSuiteResult:
    """Run the full end-to-end forensic OCR & claim verification benchmark."""
    pipeline = VeriDocPipeline(
        analyzer=analyzer or DocumentForensicAnalyzer(),
        ocr_backend=ocr_backend or TesseractExtractor(),
    )
    suite = generate_demo_suite()

    rows: list[ScenarioBenchmark] = []
    total_claims_correct = 0
    total_claims_count = 0
    tamper_correct = 0

    t_all_start = time.perf_counter()
    for scenario_id, scenario in suite.scenarios.items():
        t0 = time.perf_counter()
        audit = pipeline.audit_image(
            image_input=scenario.image_bytes,
            corpus=scenario.corpus_jsonl,
            scenario_id=scenario_id,
            include_raw_ocr=True,
            include_previews=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_claim_text = "\n".join(split_claims(audit.raw_ocr.text)) if audit.raw_ocr else ""
        restored_claim_text = "\n".join(split_claims(audit.restored_ocr.text))

        raw_char_acc = compute_char_accuracy(raw_claim_text, scenario.ground_truth_text)
        restored_char_acc = compute_char_accuracy(restored_claim_text, scenario.ground_truth_text)
        raw_word_acc = compute_word_accuracy(raw_claim_text, scenario.ground_truth_text)
        restored_word_acc = compute_word_accuracy(restored_claim_text, scenario.ground_truth_text)

        expected_verdicts = [v for _, v in scenario.expected_claims]
        actual_verdicts = [c.verdict for c in audit.verification.claims]
        matched = sum(
            1 for exp, act in zip(expected_verdicts, actual_verdicts) if exp == act
        )
        total_expected = len(expected_verdicts)
        claim_acc = matched / max(total_expected, 1)

        total_claims_correct += matched
        total_claims_count += total_expected

        tamper_detected = bool(audit.forensics and audit.forensics.tamper_flagged)
        if tamper_detected == scenario.expected_tamper:
            tamper_correct += 1

        rows.append(
            ScenarioBenchmark(
                scenario_id=scenario_id,
                raw_char_acc=round(raw_char_acc, 4),
                restored_char_acc=round(restored_char_acc, 4),
                raw_word_acc=round(raw_word_acc, 4),
                restored_word_acc=round(restored_word_acc, 4),
                raw_ocr_conf=round(audit.raw_ocr.mean_confidence if audit.raw_ocr else 0.0, 2),
                restored_ocr_conf=round(audit.restored_ocr.mean_confidence, 2),
                tamper_expected=scenario.expected_tamper,
                tamper_detected=tamper_detected,
                tamper_score=round(audit.forensics.tamper_suspicion_score if audit.forensics else 0.0, 3),
                claim_accuracy=round(claim_acc, 4),
                claims_matched=matched,
                claims_total=total_expected,
                faithfulness_score=audit.verification.faithfulness_score,
                latency_ms=round(elapsed_ms, 1),
            )
        )

    total_ms = (time.perf_counter() - t_all_start) * 1000.0
    n = max(len(rows), 1)
    return BenchmarkSuiteResult(
        scenarios=rows,
        mean_raw_char_acc=round(sum(r.raw_char_acc for r in rows) / n, 4),
        mean_restored_char_acc=round(sum(r.restored_char_acc for r in rows) / n, 4),
        mean_raw_word_acc=round(sum(r.raw_word_acc for r in rows) / n, 4),
        mean_restored_word_acc=round(sum(r.restored_word_acc for r in rows) / n, 4),
        overall_claim_accuracy=round(total_claims_correct / max(total_claims_count, 1), 4),
        tamper_detection_accuracy=round(tamper_correct / n, 4),
        total_latency_ms=round(total_ms, 1),
    )


def format_benchmark_table(result: BenchmarkSuiteResult) -> str:
    """Format benchmark results into a crisp Markdown/ASCII summary table."""
    lines = [
        "VeriDoc Forensics — End-to-End Benchmark Summary",
        "=" * 116,
        f"{'Scenario':<22} | {'Raw Char':>8} | {'Rest Char':>9} | {'Raw Word':>8} | {'Rest Word':>9} | {'Tamper':>11} | {'Claims':>7} | {'Faithful':>8} | {'Latency':>8}",
        "-" * 116,
    ]
    for r in result.scenarios:
        tamper_str = f"{'FLAG' if r.tamper_detected else 'CLEAN'}({r.tamper_score:.2f})"
        claims_str = f"{r.claims_matched}/{r.claims_total}"
        lines.append(
            f"{r.scenario_id:<22} | "
            f"{r.raw_char_acc:>8.1%} | "
            f"{r.restored_char_acc:>9.1%} | "
            f"{r.raw_word_acc:>8.1%} | "
            f"{r.restored_word_acc:>9.1%} | "
            f"{tamper_str:>11} | "
            f"{claims_str:>7} | "
            f"{r.faithfulness_score:>8.1%} | "
            f"{r.latency_ms:>6.0f}ms"
        )
    lines.extend(
        [
            "-" * 116,
            f"Mean Raw vs Restored Char Accuracy : {result.mean_raw_char_acc:.1%} -> {result.mean_restored_char_acc:.1%}",
            f"Mean Raw vs Restored Word Accuracy : {result.mean_raw_word_acc:.1%} -> {result.mean_restored_word_acc:.1%}",
            f"Claim Verdict Classification Acc.  : {result.overall_claim_accuracy:.1%}",
            f"Tamper Detection Accuracy          : {result.tamper_detection_accuracy:.1%}",
            f"Total Suite Runtime                : {result.total_latency_ms:.1f} ms",
        ]
    )
    return "\n".join(lines)
