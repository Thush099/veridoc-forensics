"""FastAPI backend and zero-build dark-mode Document Forensics & Claim Verification Studio."""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Literal
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from veridoc.models import AuditResult
from veridoc.ocr import check_tesseract_available
from veridoc.pipeline import VeriDocPipeline, render_audit_html
from veridoc.synthetic import get_reference_corpus_jsonl


class AnalyzeDemoRequest(BaseModel):
    """Request payload for running a built-in synthetic demo scenario."""

    scenario: str = "clean"


def _parse_multipart_form(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, bytes]]:
    """Binary-safe RFC 7578 multipart/form-data parser using Python stdlib only."""
    match = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', content_type, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Missing multipart boundary in Content-Type header.")
    boundary = (match.group(1) or match.group(2)).encode("latin-1")
    delimiter = b"--" + boundary

    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}

    for raw_part in body.split(delimiter)[1:]:
        if raw_part.startswith(b"--"):
            break
        part = raw_part
        if part.startswith(b"\r\n"):
            part = part[2:]
        elif part.startswith(b"\n"):
            part = part[1:]

        sep_idx = part.find(b"\r\n\r\n")
        sep_len = 4
        if sep_idx == -1:
            sep_idx = part.find(b"\n\n")
            sep_len = 2
        if sep_idx == -1:
            continue

        header_bytes = part[:sep_idx]
        payload = part[sep_idx + sep_len :]
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        elif payload.endswith(b"\n"):
            payload = payload[:-1]

        header_text = header_bytes.decode("utf-8", errors="replace")
        disp_match = re.search(r'Content-Disposition:\s*form-data;([^\r\n]+)', header_text, flags=re.IGNORECASE)
        if not disp_match:
            continue
        disp = disp_match.group(1)
        name_match = re.search(r'name="([^"]+)"', disp)
        if not name_match:
            continue
        field_name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', disp)

        if filename_match and filename_match.group(1):
            files[field_name] = payload
        elif filename_match is not None and not filename_match.group(1):
            # Empty file input field selected in browser form.
            continue
        else:
            fields[field_name] = payload.decode("utf-8", errors="replace")

    return fields, files


def _build_studio_html(default_corpus_jsonl: str) -> str:
    corpus_js = json.dumps(default_corpus_jsonl)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>VeriDoc Forensics — Multimodal OCR &amp; Claim Verification Studio</title>
  <style>
    :root {{
      --bg: #080c14;
      --surface: #111827;
      --surface-2: #0d131f;
      --border: #1e293b;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --cyan: #38bdf8;
      --emerald: #10b981;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --violet: #a855f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top right, rgba(56,189,248,0.08), transparent 45%), var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      line-height: 1.5;
    }}
    header {{
      border-bottom: 1px solid var(--border);
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(10px);
      padding: 18px 28px;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    .header-inner {{
      max-width: 1320px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .brand h1 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .brand span.tag {{
      font-size: 11px;
      font-family: ui-monospace, monospace;
      color: var(--cyan);
      border: 1px solid rgba(56,189,248,0.35);
      background: rgba(56,189,248,0.1);
      padding: 2px 8px;
      border-radius: 999px;
    }}
    .brand p {{ margin: 3px 0 0; font-size: 13px; color: var(--muted); }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px 28px 48px;
      display: grid;
      gap: 20px;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
    }}
    .card h2 {{
      margin: 0 0 14px;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }}
    .controls-grid {{
      display: grid;
      grid-template-columns: 1.1fr 1.4fr;
      gap: 20px;
    }}
    @media (max-width: 960px) {{
      .controls-grid {{ grid-template-columns: 1fr; }}
    }}
    .scenario-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }}
    button {{
      cursor: pointer;
      border-radius: 9px;
      border: 1px solid var(--border);
      background: var(--surface-2);
      color: var(--text);
      padding: 9px 14px;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.15s ease;
    }}
    button:hover {{ border-color: var(--cyan); color: #ffffff; }}
    button.active {{
      border-color: var(--cyan);
      background: rgba(56,189,248,0.16);
      color: #ffffff;
    }}
    button.primary {{
      background: linear-gradient(135deg, #0284c7, #0ea5e9);
      border-color: #38bdf8;
      color: #ffffff;
    }}
    button.primary:hover {{ filter: brightness(1.08); }}
    button:disabled {{ opacity: 0.55; cursor: not-allowed; }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 600;
    }}
    input[type="file"], textarea {{
      width: 100%;
      border-radius: 9px;
      border: 1px solid var(--border);
      background: var(--surface-2);
      color: var(--text);
      padding: 10px;
      font-family: ui-monospace, monospace;
      font-size: 12px;
    }}
    textarea {{ min-height: 118px; resize: vertical; }}
    .banner {{
      padding: 16px 20px;
      border-radius: 12px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-left: 5px solid var(--cyan);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      border: 1px solid currentColor;
      font-family: ui-monospace, monospace;
    }}
    .metrics-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }}
    .gauge-card {{
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
    }}
    .gauge-card .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; }}
    .gauge-card .val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    .gauge-bar {{
      height: 6px;
      border-radius: 999px;
      background: #1e293b;
      margin-top: 8px;
      overflow: hidden;
    }}
    .gauge-fill {{ height: 100%; border-radius: 999px; transition: width 0.25s ease; }}
    .images-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .img-panel {{
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
    }}
    .img-panel h3 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); }}
    .img-panel img {{
      width: 100%;
      max-height: 430px;
      object-fit: contain;
      border-radius: 6px;
      background: #05080f;
      display: block;
    }}
    .ocr-box {{
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      font-family: ui-monospace, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      max-height: 220px;
      overflow-y: auto;
    }}
    .chip-list {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
    .chip {{
      font-family: ui-monospace, monospace;
      font-size: 11px;
      background: rgba(245,158,11,0.12);
      color: var(--amber);
      border: 1px solid rgba(245,158,11,0.3);
      padding: 2px 8px;
      border-radius: 6px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      text-align: left;
      vertical-align: top;
      padding: 12px 10px;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
    }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    code {{
      font-family: ui-monospace, monospace;
      font-size: 12px;
      background: var(--surface-2);
      padding: 2px 6px;
      border-radius: 5px;
    }}
    .status-line {{ font-size: 12px; color: var(--cyan); font-family: ui-monospace, monospace; }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div class="brand">
        <h1>VeriDoc Forensics <span class="tag">CV + OCR + BM25/TF-IDF</span></h1>
        <p>Multimodal Document Quality Auditor, ELA Tamper Detector &amp; Citation-Grounded Claim Verifier</p>
      </div>
      <div style="display:flex;gap:10px;align-items:center">
        <span id="statusText" class="status-line">Ready</span>
        <button id="exportBtn" disabled>Export HTML Report</button>
      </div>
    </div>
  </header>

  <main>
    <section class="card">
      <h2>1. Select Synthetic Forensic Scenario or Upload Custom Document</h2>
      <div class="scenario-bar">
        <button class="scenario-btn active" data-scenario="clean">Clean Document</button>
        <button class="scenario-btn" data-scenario="skewed_blurry">Degraded Scan (Deskew + Restore)</button>
        <button class="scenario-btn" data-scenario="tampered_hallucinated">Tampered / Hallucinated Claim</button>
      </div>
      <div class="controls-grid">
        <div>
          <label for="fileInput">Upload PNG / JPG Document (optional if testing raw claim text)</label>
          <input id="fileInput" type="file" accept="image/png,image/jpeg,image/webp"/>
          <label for="claimTextInput" style="margin-top:12px">Direct Claim Text Override (optional)</label>
          <textarea id="claimTextInput" placeholder="Leave blank to run OCR on uploaded image, or paste claims directly..."></textarea>
        </div>
        <div>
          <label for="corpusInput">Trusted Reference Knowledge Base (JSONL)</label>
          <textarea id="corpusInput" style="min-height:178px"></textarea>
          <div style="margin-top:10px;display:flex;gap:10px;justify-content:flex-end">
            <button id="resetCorpusBtn" type="button">Reset Default KB</button>
            <button id="verifyCustomBtn" class="primary" type="button">Run Forensic Verify</button>
          </div>
        </div>
      </div>
    </section>

    <div id="riskBanner" class="banner">
      <div>
        <strong id="bannerTitle">Loading initial audit...</strong>
        <div id="bannerSub" style="font-size:13px;color:var(--muted);margin-top:2px"></div>
      </div>
      <span id="riskPill" class="pill" style="color:var(--cyan)">PENDING</span>
    </div>

    <section class="card" id="forensicsCard">
      <h2>2. Forensic Image Quality &amp; Tamper Telemetry</h2>
      <div class="metrics-row" id="forensicMetrics"></div>
      <div id="issuesWrap" style="margin-top:12px;font-size:13px;color:var(--muted)"></div>
      <div class="images-grid" id="imagesGrid" style="margin-top:16px"></div>
    </section>

    <section class="card">
      <h2>3. OCR Extraction &amp; Word Confidence Audit</h2>
      <div id="ocrStats" style="font-size:13px;color:var(--muted);margin-bottom:10px"></div>
      <div class="ocr-box" id="ocrText"></div>
      <div id="lowConfWrap" class="chip-list"></div>
    </section>

    <section class="card">
      <h2>4. Citation-Grounded Claim Verification &amp; Evidence Diffs</h2>
      <div id="verifSummary" style="font-size:13px;color:var(--muted);margin-bottom:12px"></div>
      <div style="overflow-x:auto">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Verdict</th>
              <th>Extracted Claim &amp; Numeric/Polarity Rationale</th>
              <th>Matched Evidence Passage</th>
            </tr>
          </thead>
          <tbody id="claimsBody"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const DEFAULT_CORPUS = {corpus_js};
    const corpusEl = document.getElementById("corpusInput");
    const fileEl = document.getElementById("fileInput");
    const claimTextEl = document.getElementById("claimTextInput");
    const statusEl = document.getElementById("statusText");
    const exportBtn = document.getElementById("exportBtn");
    let latestAudit = null;

    corpusEl.value = DEFAULT_CORPUS;

    const COLORS = {{
      LOW: "#10b981",
      MODERATE: "#f59e0b",
      HIGH: "#f97316",
      CRITICAL: "#f43f5e",
      SUPPORTED: "#10b981",
      PARTIAL: "#f59e0b",
      CONTRADICTED: "#f43f5e",
      UNVERIFIED: "#a855f7",
    }};

    function esc(s) {{
      return String(s ?? "").replace(/[&<>"']/g, c => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[c]));
    }}

    function gauge(label, valueText, pct, color) {{
      const clamped = Math.max(0, Math.min(100, pct));
      return `
        <div class="gauge-card">
          <div class="label">${{esc(label)}}</div>
          <div class="val" style="color:${{color}}">${{esc(valueText)}}</div>
          <div class="gauge-bar"><div class="gauge-fill" style="width:${{clamped}}%;background:${{color}}"></div></div>
        </div>`;
    }}

    function renderAudit(audit) {{
      latestAudit = audit;
      exportBtn.disabled = false;

      const riskColor = COLORS[audit.overall_risk] || "#38bdf8";
      document.getElementById("riskBanner").style.borderLeftColor = riskColor;
      document.getElementById("bannerTitle").textContent = audit.summary_banner;
      document.getElementById("bannerSub").textContent =
        `Scenario: ${{audit.scenario_id || "custom"}} | Faithfulness: ${{(audit.verification.faithfulness_score * 100).toFixed(1)}}% | Audited: ${{audit.created_at}}`;
      const riskPill = document.getElementById("riskPill");
      riskPill.textContent = `${{audit.overall_risk}} RISK`;
      riskPill.style.color = riskColor;

      const f = audit.forensics;
      const fCard = document.getElementById("forensicsCard");
      if (!f) {{
        fCard.style.display = "none";
      }} else {{
        fCard.style.display = "block";
        const qColor = f.quality_score >= 75 ? COLORS.LOW : (f.quality_score >= 50 ? COLORS.MODERATE : COLORS.CRITICAL);
        const tColor = f.tamper_flagged ? COLORS.CRITICAL : COLORS.LOW;
        document.getElementById("forensicMetrics").innerHTML = [
          gauge("Quality Score", `${{f.quality_score.toFixed(1)}} / 100`, f.quality_score, qColor),
          gauge("Tamper Suspicion", `${{f.tamper_suspicion_score.toFixed(3)}} (${{f.tamper_flagged ? "ALERT" : "OK"}})`, f.tamper_suspicion_score * 100, tColor),
          gauge("Blur Sharpness", `${{f.blur_laplacian_var.toFixed(0)}}`, Math.min(100, f.blur_laplacian_var / 12), f.is_blurry ? COLORS.MODERATE : COLORS.LOW),
          gauge("Skew Angle", `${{f.skew_angle_deg >= 0 ? "+" : ""}}${{f.skew_angle_deg.toFixed(2)}}°`, Math.min(100, Math.abs(f.skew_angle_deg) * 8), f.needs_deskew ? COLORS.MODERATE : COLORS.LOW),
          gauge("Dynamic Range", `${{f.dynamic_range.toFixed(0)}}`, (f.dynamic_range / 255) * 100, f.contrast_adequate ? COLORS.LOW : COLORS.MODERATE),
          gauge("Faithfulness", `${{(audit.verification.faithfulness_score * 100).toFixed(0)}}%`, audit.verification.faithfulness_score * 100, COLORS[audit.overall_risk] || "#38bdf8"),
        ].join("");

        const restText = f.restoration_applied.length ? f.restoration_applied.join(", ") : "none required";
        const issuesHtml = f.issues.map(i => `<code>${{esc(i.code)}}</code> ${{esc(i.message)}}`).join(" &bull; ");
        document.getElementById("issuesWrap").innerHTML =
          `<strong>Restorations:</strong> <code>${{esc(restText)}}</code> &bull; <strong>Effective DPI:</strong> ${{f.estimated_dpi}} &bull; ${{issuesHtml}}`;

        const imgs = [];
        if (f.raw_image_base64) {{
          imgs.push(`<div class="img-panel"><h3>Raw Input Scan</h3><img src="${{f.raw_image_base64}}" alt="Raw scan"/></div>`);
        }}
        if (f.restored_image_base64) {{
          imgs.push(`<div class="img-panel"><h3>Forensic-Restored Output</h3><img src="${{f.restored_image_base64}}" alt="Restored scan"/></div>`);
        }}
        if (f.ela_heatmap_base64) {{
          imgs.push(`<div class="img-panel"><h3>ELA &amp; Noise Tamper Heatmap (${{f.suspicious_regions.length}} flagged tile(s))</h3><img src="${{f.ela_heatmap_base64}}" alt="ELA heatmap"/></div>`);
        }}
        document.getElementById("imagesGrid").innerHTML = imgs.join("");
      }}

      const rawStats = audit.raw_ocr
        ? `Raw OCR: <strong>${{audit.raw_ocr.word_count}}</strong> words (${{audit.raw_ocr.mean_confidence.toFixed(1)}}% conf) &rarr; `
        : "";
      document.getElementById("ocrStats").innerHTML =
        `${{rawStats}}Restored OCR (<code>${{esc(audit.restored_ocr.engine)}}</code>): <strong>${{audit.restored_ocr.word_count}}</strong> words &bull; Mean confidence: <strong>${{audit.restored_ocr.mean_confidence.toFixed(1)}}%</strong>`;
      document.getElementById("ocrText").textContent = audit.restored_ocr.text || "(No text extracted)";
      document.getElementById("lowConfWrap").innerHTML = audit.restored_ocr.low_confidence_words.length
        ? `<span style="font-size:12px;color:var(--muted)">Low-conf tokens (&lt;70%):</span>` +
          audit.restored_ocr.low_confidence_words.slice(0, 20).map(w => `<span class="chip">${{esc(w)}}</span>`).join("")
        : "";

      const v = audit.verification;
      document.getElementById("verifSummary").innerHTML =
        `Total Claims: <strong>${{v.total_claims}}</strong> &bull; ` +
        `<span style="color:var(--emerald)">Supported: ${{v.supported_count}}</span> &bull; ` +
        `<span style="color:var(--amber)">Partial: ${{v.partial_count}}</span> &bull; ` +
        `<span style="color:var(--rose)">Contradicted: ${{v.contradicted_count}}</span> &bull; ` +
        `<span style="color:var(--violet)">Unverified: ${{v.unverified_count}}</span>`;

      const rows = v.claims.map(c => {{
        const vc = COLORS[c.verdict] || "#94a3b8";
        const top = c.matched_passages && c.matched_passages[0];
        const evHtml = top
          ? `<code>${{esc(top.passage_id)}}</code> <small style="color:var(--muted)">(hybrid ${{top.hybrid_score.toFixed(2)}})</small><div style="margin-top:4px">${{esc(top.text)}}</div>`
          : `<span style="color:var(--muted)">No matched evidence</span>`;
        const numsDiff = `Claim nums: <code>${{esc(c.extracted_numbers.join(", ") || "none")}}</code> vs Evidence nums: <code>${{esc(c.evidence_numbers.join(", ") || "none")}}</code>`;
        return `
          <tr>
            <td><code>${{esc(c.claim_id)}}</code></td>
            <td><span class="pill" style="color:${{vc}}">${{esc(c.verdict)}}</span><div style="font-size:11px;color:var(--muted);margin-top:4px">${{(c.confidence * 100).toFixed(0)}}% conf</div></td>
            <td><strong>${{esc(c.claim_text)}}</strong><div style="font-size:12px;color:var(--muted);margin-top:4px">${{esc(c.rationale)}}</div><div style="font-size:11px;color:var(--muted);margin-top:4px">${{numsDiff}}</div></td>
            <td>${{evHtml}}</td>
          </tr>`;
      }}).join("");
      document.getElementById("claimsBody").innerHTML = rows || `<tr><td colspan="4" style="color:var(--muted)">No claims extracted.</td></tr>`;
    }}

    async function runDemo(scenario) {{
      statusEl.textContent = `Running ${{scenario}}...`;
      document.querySelectorAll(".scenario-btn").forEach(b => b.classList.toggle("active", b.dataset.scenario === scenario));
      try {{
        const res = await fetch("/api/analyze-demo", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ scenario }})
        }});
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        renderAudit(data);
        statusEl.textContent = `Completed: ${{scenario}}`;
      }} catch (err) {{
        statusEl.textContent = `Error: ${{err.message}}`;
      }}
    }}

    document.querySelectorAll(".scenario-btn").forEach(btn => {{
      btn.addEventListener("click", () => runDemo(btn.dataset.scenario));
    }});

    document.getElementById("resetCorpusBtn").addEventListener("click", () => {{
      corpusEl.value = DEFAULT_CORPUS;
    }});

    document.getElementById("verifyCustomBtn").addEventListener("click", async () => {{
      const fd = new FormData();
      if (fileEl.files && fileEl.files[0]) {{
        fd.append("file", fileEl.files[0]);
      }}
      if (claimTextEl.value.trim()) {{
        fd.append("claim_text", claimTextEl.value.trim());
      }}
      fd.append("corpus_jsonl", corpusEl.value);
      statusEl.textContent = "Running custom forensic verification...";
      try {{
        const res = await fetch("/api/verify", {{ method: "POST", body: fd }});
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        renderAudit(data);
        statusEl.textContent = "Custom verification complete";
      }} catch (err) {{
        statusEl.textContent = `Error: ${{err.message}}`;
      }}
    }});

    exportBtn.addEventListener("click", async () => {{
      if (!latestAudit) return;
      const res = await fetch("/api/export-html", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(latestAudit)
      }});
      const htmlText = await res.text();
      const blob = new Blob([htmlText], {{ type: "text/html;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `veridoc-audit-${{latestAudit.scenario_id || "custom"}}.html`;
      a.click();
      URL.revokeObjectURL(url);
    }});

    runDemo("clean");
  </script>
</body>
</html>
"""


def create_app(pipeline: VeriDocPipeline | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    svc = pipeline or VeriDocPipeline()
    application = FastAPI(
        title="VeriDoc Forensics",
        description="Multimodal Document OCR, Forensic Image Quality Auditor & Citation-Grounded Claim Verification Studio",
        version="0.1.0",
    )

    @application.get("/", response_class=HTMLResponse)
    async def studio_ui() -> HTMLResponse:
        return HTMLResponse(content=_build_studio_html(get_reference_corpus_jsonl()))

    @application.get("/health")
    async def health() -> dict[str, Any]:
        available, version = check_tesseract_available()
        return {
            "status": "ok",
            "service": "veridoc-forensics",
            "version": "0.1.0",
            "tesseract_available": available,
            "tesseract_version": version,
        }

    @application.post("/api/analyze-demo", response_model=AuditResult)
    async def analyze_demo(request: Request, scenario: str | None = None) -> AuditResult:
        selected = scenario
        if selected is None:
            raw_body = await request.body()
            if raw_body.strip():
                try:
                    payload = AnalyzeDemoRequest.model_validate_json(raw_body)
                    selected = payload.scenario
                except ValidationError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
            else:
                selected = "clean"

        try:
            return svc.audit_demo(selected, include_previews=True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/verify", response_model=AuditResult)
    async def verify_document(request: Request) -> AuditResult:
        content_type = request.headers.get("content-type", "")
        body = await request.body()

        file_bytes: bytes | None = None
        claim_text: str | None = None
        corpus_jsonl: str | None = None

        if "multipart/form-data" in content_type.lower():
            try:
                fields, files = _parse_multipart_form(body, content_type)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            file_bytes = files.get("file") or (next(iter(files.values())) if files else None)
            claim_text = fields.get("claim_text")
            corpus_jsonl = fields.get("corpus_jsonl")
        elif "application/x-www-form-urlencoded" in content_type.lower():
            parsed_qs = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
            claim_text = parsed_qs.get("claim_text", [None])[0]
            corpus_jsonl = parsed_qs.get("corpus_jsonl", [None])[0]
        else:
            if not body.strip():
                raise HTTPException(status_code=400, detail="Provide multipart form data, URL-encoded form data, or JSON payload.")
            try:
                data = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc
            if not isinstance(data, dict):
                raise HTTPException(status_code=400, detail="JSON body must be an object.")
            img_b64 = data.get("image_base64")
            if isinstance(img_b64, str) and img_b64.strip():
                raw_b64 = img_b64.split(",", 1)[1] if "," in img_b64 else img_b64
                try:
                    file_bytes = base64.b64decode(raw_b64)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc
            claim_text = data.get("claim_text")
            corpus_jsonl = data.get("corpus_jsonl")

        corpus_arg = corpus_jsonl.strip() if isinstance(corpus_jsonl, str) and corpus_jsonl.strip() else None

        if file_bytes:
            try:
                return svc.audit_image(
                    image_input=file_bytes,
                    corpus=corpus_arg,
                    scenario_id="uploaded_image",
                    include_raw_ocr=True,
                    include_previews=True,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        if isinstance(claim_text, str) and claim_text.strip():
            try:
                return svc.audit_text(
                    claim_text=claim_text,
                    corpus=corpus_arg,
                    scenario_id="direct_claim_text",
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        raise HTTPException(
            status_code=400,
            detail="Request must include either an uploaded image file ('file'/'image_base64') or non-empty 'claim_text'.",
        )

    @application.post("/api/export-html")
    async def export_html(request: Request, theme: Literal["dark", "light"] = "dark") -> Response:
        raw_body = await request.body()
        if not raw_body.strip():
            raise HTTPException(status_code=400, detail="Request body must contain an AuditResult JSON payload.")
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

        req_theme: Literal["dark", "light"] = theme
        audit_payload = parsed
        if isinstance(parsed, dict) and "audit" in parsed:
            audit_payload = parsed["audit"]
            if parsed.get("theme") in {"dark", "light"}:
                req_theme = parsed["theme"]

        try:
            audit = AuditResult.model_validate(audit_payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        rendered = render_audit_html(audit, theme=req_theme)
        return HTMLResponse(
            content=rendered,
            headers={"Content-Disposition": 'attachment; filename="veridoc-audit-report.html"'},
        )

    @application.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return application


app = create_app()
