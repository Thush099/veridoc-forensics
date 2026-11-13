"""Tests for Tesseract OCR wrapper, pluggable OCRBackend, claim splitting, and hybrid verifier."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from veridoc.forensics import DocumentForensicAnalyzer
from veridoc.models import OCRResult
from veridoc.ocr import OCRBackend, TesseractExtractor, check_tesseract_available
from veridoc.pipeline import VeriDocPipeline
from veridoc.synthetic import generate_demo_suite, get_reference_corpus_jsonl
from veridoc.verifier import ClaimVerifier, HybridEvidenceRetriever, parse_corpus_jsonl, split_claims


def test_tesseract_available_and_extracts_clean_document() -> None:
    available, version = check_tesseract_available()
    assert available is True
    assert version is not None

    suite = generate_demo_suite()
    extractor = TesseractExtractor()
    assert isinstance(extractor, OCRBackend)

    result = extractor.extract(suite.scenarios["clean"].image_bytes)
    assert result.engine == "tesseract"
    assert result.word_count >= 80
    assert result.mean_confidence >= 85.0
    assert "$482,500" in result.text
    assert "99.94%" in result.text
    assert "FIN-402" in result.text


def test_tesseract_handles_blank_image_gracefully() -> None:
    blank = Image.new("RGB", (400, 200), color=(255, 255, 255))
    buf = io.BytesIO()
    blank.save(buf, format="PNG")

    result = TesseractExtractor().extract(buf.getvalue())
    assert result.text == ""
    assert result.word_count == 0
    assert result.mean_confidence == 0.0
    assert result.low_confidence_words == []


def test_restoration_dramatically_improves_degraded_ocr() -> None:
    suite = generate_demo_suite()
    extractor = TesseractExtractor()
    analyzer = DocumentForensicAnalyzer()
    skewed_bytes = suite.scenarios["skewed_blurry"].image_bytes

    raw_ocr = extractor.extract(skewed_bytes)
    restored_bytes, _ = analyzer.analyze_and_restore(skewed_bytes, include_previews=False)
    restored_ocr = extractor.extract(restored_bytes)

    assert restored_ocr.word_count > raw_ocr.word_count
    assert restored_ocr.mean_confidence > raw_ocr.mean_confidence
    assert "$482,500" in restored_ocr.text
    assert "99.94%" in restored_ocr.text


def test_split_claims_preserves_decimals_and_drops_header_chrome() -> None:
    sample = (
        "APEX MERIDIAN SYSTEMS\n"
        "Q3 2026 Regulatory Compliance & Settlement Summary\n"
        "DOCUMENT ID: AMS-2026-Q3\n"
        "1. Acme Corp. reported $12,450.50 in Q3 2026 expenses across U.S. regions.\n"
        "2. Availability hit 99.94%.\n"
        "3. Backup retention stayed at 7 years.\n"
        "CONFIDENTIAL COMPLIANCE FILING\n"
    )
    claims = split_claims(sample)
    assert len(claims) == 3
    assert "$12,450.50" in claims[0]
    assert "U.S. regions." in claims[0]
    assert "99.94%" in claims[1]
    assert "7 years." in claims[2]


def test_hybrid_verifier_discriminates_supported_partial_contradicted_unverified() -> None:
    corpus_jsonl = get_reference_corpus_jsonl()
    verifier = ClaimVerifier(corpus_jsonl)

    claims = [
        "Apex Meridian Systems reported total Q3 2026 operating settlement expenses of $482,500 across 14 active cloud regions.",
        "Apex Meridian Systems reported total Q3 2026 operating settlement expenses of $940,000 across 14 active cloud regions.",
        "The independent SOC 2 Type II compliance audit failed on October 14, 2026 with 6 critical security findings.",
        "All financial ledger backups are encrypted using AES-256-GCM under Policy FIN-402 and audited quarterly.",
        "The Zurich satellite clearing hub processes 12,000 daily Swiss franc derivatives contracts.",
    ]
    summary = verifier.verify_claims(claims)

    assert summary.total_claims == 5
    assert summary.claims[0].verdict == "SUPPORTED"
    assert summary.claims[0].matched_passages[0].passage_id == "KB-001"
    assert summary.claims[1].verdict == "CONTRADICTED"
    assert summary.claims[2].verdict == "CONTRADICTED"
    assert summary.claims[3].verdict in {"PARTIAL", "SUPPORTED"}
    assert summary.claims[4].verdict == "UNVERIFIED"
    assert summary.contradicted_count == 2
    assert summary.unverified_count == 1
    assert 0.0 < summary.faithfulness_score < 0.5


def test_negation_flip_contradiction_and_retriever_small_corpus() -> None:
    mini_corpus = json.dumps(
        {
            "id": "C-1",
            "title": "Encryption",
            "text": "Customer vault backups are encrypted with AES-256 before archival.",
        }
    )
    passages = parse_corpus_jsonl(mini_corpus)
    retriever = HybridEvidenceRetriever(passages)
    hits = retriever.retrieve("Customer vault backups are not encrypted before archival.")
    assert len(hits) == 1 and hits[0].passage_id == "C-1"

    summary = ClaimVerifier(passages).verify_claims(
        ["Customer vault backups are not encrypted with AES-256 before archival."]
    )
    assert summary.claims[0].verdict == "CONTRADICTED"
    assert summary.claims[0].negation_mismatch is True


def test_parse_corpus_jsonl_validation_and_pluggable_backend() -> None:
    with pytest.raises(ValueError):
        parse_corpus_jsonl("{not-valid-json")
    with pytest.raises(ValueError):
        parse_corpus_jsonl(json.dumps({"id": "X"}))

    class FakeOCR:
        def extract(self, image_input: object) -> OCRResult:
            return OCRResult(
                text="All financial ledger backups are encrypted using AES-256-GCM and retained for 7 years under Policy FIN-402.",
                words=[],
                word_count=16,
                mean_confidence=99.0,
                low_confidence_words=[],
                engine="fake-ocr",
            )

    suite = generate_demo_suite()
    pipeline = VeriDocPipeline(ocr_backend=FakeOCR())
    audit = pipeline.audit_image(suite.scenarios["clean"].image_bytes, corpus=suite.corpus_jsonl, include_previews=False)
    assert audit.restored_ocr.engine == "fake-ocr"
    assert audit.verification.supported_count == 1
    assert audit.verification.faithfulness_score == 1.0
