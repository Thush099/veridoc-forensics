"""VeriDoc Forensics — Multimodal Document OCR, Image Forensics & Claim Verification."""

from veridoc.benchmark import BenchmarkSuiteResult, run_benchmark
from veridoc.forensics import DocumentForensicAnalyzer, build_before_after_svg, encode_image_base64
from veridoc.models import AuditResult, ClaimAudit, ForensicReport, OCRResult, VerificationSummary
from veridoc.ocr import OCRBackend, TesseractExtractor, check_tesseract_available
from veridoc.pipeline import VeriDocPipeline, render_audit_html
from veridoc.synthetic import DemoScenario, DemoSuite, generate_demo_suite, get_reference_corpus_jsonl
from veridoc.verifier import ClaimVerifier, HybridEvidenceRetriever, parse_corpus_jsonl, split_claims

__version__ = "0.1.0"

__all__ = [
    "AuditResult",
    "BenchmarkSuiteResult",
    "ClaimAudit",
    "ClaimVerifier",
    "DemoScenario",
    "DemoSuite",
    "DocumentForensicAnalyzer",
    "ForensicReport",
    "HybridEvidenceRetriever",
    "OCRBackend",
    "OCRResult",
    "TesseractExtractor",
    "VeriDocPipeline",
    "VerificationSummary",
    "__version__",
    "build_before_after_svg",
    "check_tesseract_available",
    "encode_image_base64",
    "generate_demo_suite",
    "get_reference_corpus_jsonl",
    "parse_corpus_jsonl",
    "render_audit_html",
    "run_benchmark",
    "split_claims",
]
