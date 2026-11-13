"""Pydantic schemas shared across the VeriDoc Forensics pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SeverityLevel = Literal["info", "warning", "critical"]
ClaimVerdict = Literal["SUPPORTED", "PARTIAL", "CONTRADICTED", "UNVERIFIED"]
RiskLevel = Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]


class ForensicIssue(BaseModel):
    """Single quality or tampering anomaly detected during image inspection."""

    code: str
    severity: SeverityLevel
    message: str
    metric_value: float | None = None


class SuspiciousRegion(BaseModel):
    """Bounding box for a tile exhibiting anomalous ELA or local noise shift."""

    x: int
    y: int
    width: int
    height: int
    ela_zscore: float
    noise_anomaly: float


class ForensicReport(BaseModel):
    """Structured image quality, geometric skew, and tamper forensic report."""

    width: int
    height: int
    channels: int
    estimated_dpi: float
    dpi_adequate: bool
    blur_laplacian_var: float
    is_blurry: bool
    skew_angle_deg: float
    needs_deskew: bool
    deskew_applied: bool
    mean_brightness: float
    contrast_std: float
    dynamic_range: float
    highlight_clip_ratio: float
    shadow_clip_ratio: float
    contrast_adequate: bool
    ela_mean_residual: float
    ela_max_tile_residual: float
    ela_tile_variance: float
    local_noise_anomaly_score: float
    tamper_suspicion_score: float = Field(ge=0.0, le=1.0)
    tamper_flagged: bool
    suspicious_regions: list[SuspiciousRegion] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=100.0)
    restoration_applied: list[str] = Field(default_factory=list)
    issues: list[ForensicIssue] = Field(default_factory=list)
    raw_image_base64: str | None = None
    restored_image_base64: str | None = None
    ela_heatmap_base64: str | None = None
    comparison_svg: str | None = None


class WordBox(BaseModel):
    """Single word token detected by OCR along with bounding box and confidence."""

    text: str
    confidence: float = Field(ge=0.0, le=100.0)
    left: int
    top: int
    width: int
    height: int
    line_num: int = 0
    block_num: int = 0


class OCRResult(BaseModel):
    """Structured OCR output returned by an OCR backend."""

    text: str
    words: list[WordBox] = Field(default_factory=list)
    word_count: int = 0
    mean_confidence: float = 0.0
    low_confidence_words: list[str] = Field(default_factory=list)
    engine: str = "tesseract"


class CorpusPassage(BaseModel):
    """Single reference passage from a trusted knowledge-base JSONL."""

    id: str
    title: str = ""
    text: str


class EvidenceMatch(BaseModel):
    """Ranked evidence passage returned by hybrid BM25 + TF-IDF retrieval."""

    passage_id: str
    title: str = ""
    text: str
    bm25_score: float
    tfidf_score: float
    hybrid_score: float


class ClaimAudit(BaseModel):
    """Citation-grounded entailment audit result for a single extracted claim."""

    claim_id: str
    claim_text: str
    verdict: ClaimVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    matched_passages: list[EvidenceMatch] = Field(default_factory=list)
    extracted_numbers: list[str] = Field(default_factory=list)
    evidence_numbers: list[str] = Field(default_factory=list)
    negation_mismatch: bool = False


class VerificationSummary(BaseModel):
    """Aggregate faithfulness summary across all extracted document claims."""

    total_claims: int
    supported_count: int
    partial_count: int
    contradicted_count: int
    unverified_count: int
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    claims: list[ClaimAudit] = Field(default_factory=list)


class AuditResult(BaseModel):
    """End-to-end multimodal forensics + OCR + claim verification audit payload."""

    scenario_id: str | None = None
    forensics: ForensicReport | None = None
    raw_ocr: OCRResult | None = None
    restored_ocr: OCRResult
    verification: VerificationSummary
    overall_risk: RiskLevel
    summary_banner: str
    created_at: str
