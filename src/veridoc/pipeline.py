"""End-to-end multimodal pipeline orchestrator and standalone HTML report renderer."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Literal, Sequence, Union

from veridoc.forensics import DocumentForensicAnalyzer, ImageInput
from veridoc.models import (
    AuditResult,
    CorpusPassage,
    ForensicReport,
    OCRResult,
    RiskLevel,
    VerificationSummary,
)
from veridoc.ocr import OCRBackend, TesseractExtractor
from veridoc.synthetic import DemoSuite, generate_demo_suite, get_reference_corpus_jsonl
from veridoc.verifier import ClaimVerifier


class VeriDocPipeline:
    """Runs forensic inspection -> restoration -> OCR -> citation-grounded verification."""

    def __init__(
        self,
        analyzer: DocumentForensicAnalyzer | None = None,
        ocr_backend: OCRBackend | None = None,
    ) -> None:
        self.analyzer = analyzer or DocumentForensicAnalyzer()
        self.ocr_backend = ocr_backend or TesseractExtractor()
        self._demo_cache: DemoSuite | None = None

    def get_demo_suite(self) -> DemoSuite:
        """Lazily cache synthetic demo scenarios for responsive CLI and API execution."""
        if self._demo_cache is None:
            self._demo_cache = generate_demo_suite()
        return self._demo_cache

    def audit_demo(
        self,
        scenario_id: str,
        include_previews: bool = True,
    ) -> AuditResult:
        """Run full pipeline on one of ``clean``, ``skewed_blurry``, or ``tampered_hallucinated``."""
        suite = self.get_demo_suite()
        if scenario_id not in suite.scenarios:
            valid = ", ".join(sorted(suite.scenarios))
            raise KeyError(f"Unknown demo scenario '{scenario_id}'. Valid options: {valid}")
        scenario = suite.scenarios[scenario_id]
        return self.audit_image(
            image_input=scenario.image_bytes,
            corpus=scenario.corpus_jsonl,
            scenario_id=scenario_id,
            include_raw_ocr=True,
            include_previews=include_previews,
        )

    def audit_image(
        self,
        image_input: ImageInput,
        corpus: Union[str, Sequence[dict[str, str] | CorpusPassage], None] = None,
        scenario_id: str | None = None,
        include_raw_ocr: bool = True,
        include_previews: bool = True,
    ) -> AuditResult:
        """Audit a document image with forensics, OCR, and evidence-grounded verification."""
        corpus_source = corpus if corpus is not None else get_reference_corpus_jsonl()
        restored_bytes, forensic_report = self.analyzer.analyze_and_restore(
            image_input=image_input,
            include_previews=include_previews,
        )
        raw_ocr = self.ocr_backend.extract(image_input) if include_raw_ocr else None

        # Reuse raw OCR when no restoration transformation altered the pixels.
        if raw_ocr is not None and not forensic_report.restoration_applied:
            restored_ocr = raw_ocr
        else:
            restored_ocr = self.ocr_backend.extract(restored_bytes)

        verifier = ClaimVerifier(corpus_source)
        verification = verifier.verify_document(restored_ocr.text)

        risk, banner = _summarize_risk(forensic_report, verification)
        return AuditResult(
            scenario_id=scenario_id,
            forensics=forensic_report,
            raw_ocr=raw_ocr,
            restored_ocr=restored_ocr,
            verification=verification,
            overall_risk=risk,
            summary_banner=banner,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def audit_text(
        self,
        claim_text: str,
        corpus: Union[str, Sequence[dict[str, str] | CorpusPassage], None] = None,
        scenario_id: str | None = None,
    ) -> AuditResult:
        """Audit raw text claims directly against a knowledge-base corpus without an image."""
        corpus_source = corpus if corpus is not None else get_reference_corpus_jsonl()
        verifier = ClaimVerifier(corpus_source)
        verification = verifier.verify_document(claim_text)

        synthetic_ocr = OCRResult(
            text=claim_text.strip(),
            words=[],
            word_count=len(claim_text.split()),
            mean_confidence=100.0 if claim_text.strip() else 0.0,
            low_confidence_words=[],
            engine="direct-text",
        )
        risk, banner = _summarize_risk(None, verification)
        return AuditResult(
            scenario_id=scenario_id,
            forensics=None,
            raw_ocr=None,
            restored_ocr=synthetic_ocr,
            verification=verification,
            overall_risk=risk,
            summary_banner=banner,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def _summarize_risk(
    forensics: ForensicReport | None,
    verification: VerificationSummary,
) -> tuple[RiskLevel, str]:
    tamper = bool(forensics and forensics.tamper_flagged)
    contradicted = verification.contradicted_count
    unverified = verification.unverified_count
    partial = verification.partial_count

    if tamper and contradicted > 0:
        return (
            "CRITICAL",
            f"CRITICAL ALERT: Splice tampering detected ({forensics.tamper_suspicion_score:.2f}) "
            f"alongside {contradicted} contradicted claim(s).",
        )
    if tamper:
        return (
            "HIGH",
            f"HIGH RISK: Image integrity anomaly detected (tamper score {forensics.tamper_suspicion_score:.2f}).",
        )
    if contradicted > 0:
        return (
            "HIGH",
            f"HIGH RISK: {contradicted} extracted claim(s) directly contradict trusted reference evidence.",
        )
    if unverified > 0 or partial > 0:
        return (
            "MODERATE",
            f"MODERATE RISK: {unverified} unverified and {partial} partial claim(s) require reviewer confirmation.",
        )
    if forensics and forensics.restoration_applied:
        return (
            "LOW",
            f"VERIFIED AFTER RESTORATION: Applied {', '.join(forensics.restoration_applied)}; "
            f"all {verification.supported_count} claims grounded ({verification.faithfulness_score:.0%}).",
        )
    return (
        "LOW",
        f"VERIFIED AUTHENTIC: {verification.supported_count}/{verification.total_claims} claims supported "
        f"({verification.faithfulness_score:.0%} faithfulness).",
    )


def render_audit_html(audit: AuditResult, theme: Literal["dark", "light"] = "dark") -> str:
    """Render a standalone printable HTML forensic audit report from an ``AuditResult``."""
    is_light = theme == "light"
    bg = "#f8fafc" if is_light else "#0b0f17"
    card = "#ffffff" if is_light else "#131a26"
    border = "#cbd5e1" if is_light else "#1f2a3d"
    text_main = "#0f172a" if is_light else "#e2e8f0"
    text_muted = "#475569" if is_light else "#94a3b8"
    code_bg = "#f1f5f9" if is_light else "#0d131f"

    risk_palette = {
        "LOW": "#10b981",
        "MODERATE": "#f59e0b",
        "HIGH": "#f97316",
        "CRITICAL": "#f43f5e",
    }
    verdict_palette = {
        "SUPPORTED": "#10b981",
        "PARTIAL": "#f59e0b",
        "CONTRADICTED": "#f43f5e",
        "UNVERIFIED": "#a855f7",
    }
    risk_color = risk_palette.get(audit.overall_risk, "#38bdf8")

    forensic_section = ""
    if audit.forensics:
        f = audit.forensics
        restorations = ", ".join(f.restoration_applied) if f.restoration_applied else "None needed"
        issue_items = "".join(
            f"<li><code>{html.escape(i.code)}</code> [{html.escape(i.severity.upper())}] — {html.escape(i.message)}</li>"
            for i in f.issues
        )
        img_grid = ""
        if f.raw_image_base64 and f.restored_image_base64:
            ela_block = (
                f'<div class="img-card"><h4>ELA Tamper Heatmap</h4>'
                f'<img src="{f.ela_heatmap_base64}" alt="ELA Heatmap"/></div>'
                if f.ela_heatmap_base64
                else ""
            )
            img_grid = (
                '<div class="img-grid">'
                f'<div class="img-card"><h4>Raw Input Scan</h4><img src="{f.raw_image_base64}" alt="Raw scan"/></div>'
                f'<div class="img-card"><h4>Forensic-Restored Output</h4><img src="{f.restored_image_base64}" alt="Restored scan"/></div>'
                f"{ela_block}"
                "</div>"
            )
        forensic_section = f"""
        <section class="card">
          <h2>Image Forensics &amp; Integrity Diagnostics</h2>
          <div class="metrics">
            <div class="metric"><span>Quality Score</span><strong>{f.quality_score:.1f} / 100</strong></div>
            <div class="metric"><span>Tamper Suspicion</span><strong style="color:{'#f43f5e' if f.tamper_flagged else '#10b981'}">{f.tamper_suspicion_score:.3f} ({'FLAGGED' if f.tamper_flagged else 'CLEAN'})</strong></div>
            <div class="metric"><span>Blur (Laplacian Var)</span><strong>{f.blur_laplacian_var:.1f}</strong></div>
            <div class="metric"><span>Skew Angle</span><strong>{f.skew_angle_deg:+.2f}°</strong></div>
            <div class="metric"><span>Dynamic Range</span><strong>{f.dynamic_range:.1f}</strong></div>
            <div class="metric"><span>Effective DPI</span><strong>{f.estimated_dpi:.1f}</strong></div>
          </div>
          <p class="muted"><strong>Restorations Applied:</strong> {html.escape(restorations)}</p>
          <ul class="issues">{issue_items}</ul>
          {img_grid}
        </section>
        """

    raw_ocr_summary = ""
    if audit.raw_ocr and audit.raw_ocr is not audit.restored_ocr:
        raw_ocr_summary = (
            f"Raw OCR: {audit.raw_ocr.word_count} words ({audit.raw_ocr.mean_confidence:.1f}% mean conf) &rarr; "
        )

    claim_rows = []
    for c in audit.verification.claims:
        v_color = verdict_palette.get(c.verdict, "#94a3b8")
        top_ev = c.matched_passages[0] if c.matched_passages else None
        ev_html = (
            f"<code>{html.escape(top_ev.passage_id)}</code> (score {top_ev.hybrid_score:.2f}): "
            f"{html.escape(top_ev.text)}"
            if top_ev
            else "No matching passage"
        )
        nums_html = (
            f"<div class='nums'>Claim numbers: <code>{html.escape(', '.join(c.extracted_numbers) or 'none')}</code> | "
            f"Evidence numbers: <code>{html.escape(', '.join(c.evidence_numbers) or 'none')}</code></div>"
        )
        claim_rows.append(
            f"""
            <tr>
              <td><code>{html.escape(c.claim_id)}</code></td>
              <td><span class="pill" style="border-color:{v_color};color:{v_color}">{html.escape(c.verdict)}</span><br/>
                  <small class="muted">{c.confidence:.0%} conf</small></td>
              <td><strong>{html.escape(c.claim_text)}</strong><br/><small class="muted">{html.escape(c.rationale)}</small>{nums_html}</td>
              <td>{ev_html}</td>
            </tr>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>VeriDoc Forensics — Audit Report</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 32px 24px; background: {bg}; color: {text_main}; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; line-height: 1.5; }}
    .wrap {{ max-width: 1120px; margin: 0 auto; }}
    .banner {{ padding: 20px 24px; border-radius: 12px; background: {card}; border: 1px solid {border}; border-left: 6px solid {risk_color}; margin-bottom: 20px; }}
    .banner h1 {{ margin: 0 0 6px; font-size: 22px; }}
    .card {{ padding: 20px 24px; border-radius: 12px; background: {card}; border: 1px solid {border}; margin-bottom: 20px; }}
    .card h2 {{ margin: 0 0 14px; font-size: 17px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 14px; }}
    .metric {{ padding: 12px; border-radius: 8px; background: {code_bg}; border: 1px solid {border}; }}
    .metric span {{ display: block; font-size: 11px; text-transform: uppercase; color: {text_muted}; }}
    .metric strong {{ font-size: 16px; }}
    .img-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-top: 14px; }}
    .img-card {{ padding: 10px; border-radius: 8px; background: {code_bg}; border: 1px solid {border}; }}
    .img-card h4 {{ margin: 0 0 8px; font-size: 13px; color: {text_muted}; }}
    .img-card img {{ width: 100%; height: auto; border-radius: 4px; display: block; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; vertical-align: top; padding: 12px 10px; border-bottom: 1px solid {border}; font-size: 13px; }}
    th {{ font-size: 11px; text-transform: uppercase; color: {text_muted}; }}
    .pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; border: 1px solid; font-size: 11px; font-weight: 700; }}
    code {{ background: {code_bg}; padding: 1px 5px; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 12px; }}
    pre {{ background: {code_bg}; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; white-space: pre-wrap; }}
    .muted {{ color: {text_muted}; }}
    .nums {{ margin-top: 4px; font-size: 12px; color: {text_muted}; }}
    ul.issues {{ margin: 8px 0 0; padding-left: 18px; font-size: 13px; }}
    @media print {{ body {{ background: #ffffff; color: #000000; padding: 10px; }} .card, .banner {{ break-inside: avoid; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="banner">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
        <div>
          <h1>VeriDoc Forensics — Compliance &amp; Claim Audit Report</h1>
          <div class="muted">Scenario: <code>{html.escape(audit.scenario_id or 'custom')}</code> &bull; Generated: {html.escape(audit.created_at)}</div>
        </div>
        <span class="pill" style="border-color:{risk_color};color:{risk_color};font-size:13px;padding:5px 14px">{html.escape(audit.overall_risk)} RISK</span>
      </div>
      <p style="margin:12px 0 0;font-weight:600">{html.escape(audit.summary_banner)}</p>
    </div>

    {forensic_section}

    <section class="card">
      <h2>OCR Extraction Telemetry</h2>
      <p class="muted">{raw_ocr_summary}Restored OCR ({html.escape(audit.restored_ocr.engine)}): <strong>{audit.restored_ocr.word_count}</strong> words &bull; Mean confidence: <strong>{audit.restored_ocr.mean_confidence:.1f}%</strong> &bull; Faithfulness score: <strong>{audit.verification.faithfulness_score:.0%}</strong></p>
      <pre>{html.escape(audit.restored_ocr.text)}</pre>
    </section>

    <section class="card">
      <h2>Citation-Grounded Claim Verification ({audit.verification.supported_count} Supported / {audit.verification.partial_count} Partial / {audit.verification.contradicted_count} Contradicted / {audit.verification.unverified_count} Unverified)</h2>
      <table>
        <thead>
          <tr><th>ID</th><th>Verdict</th><th>Extracted Claim &amp; Audit Rationale</th><th>Top Grounding Evidence</th></tr>
        </thead>
        <tbody>
          {''.join(claim_rows) or '<tr><td colspan="4" class="muted">No factual claims extracted.</td></tr>'}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""
