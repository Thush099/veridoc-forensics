"""Offline synthetic compliance document generator and reference corpus suite."""

from __future__ import annotations

import io
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from veridoc.models import ClaimVerdict

REFERENCE_PASSAGES: list[dict[str, str]] = [
    {
        "id": "KB-001",
        "title": "Q3 Operating Settlement",
        "text": "Apex Meridian Systems reported total Q3 2026 operating settlement expenses of $482,500 across 14 active cloud regions.",
    },
    {
        "id": "KB-002",
        "title": "SOC 2 Type II Audit",
        "text": "The independent SOC 2 Type II compliance audit was completed on October 14, 2026 with zero critical security findings.",
    },
    {
        "id": "KB-003",
        "title": "Ledger Backup & Retention",
        "text": "All financial ledger backups are encrypted using AES-256-GCM and retained for 7 years under Policy FIN-402.",
    },
    {
        "id": "KB-004",
        "title": "Clearing API SLA",
        "text": "Primary clearing API availability reached 99.94% during Q3 2026, exceeding the 99.90% contractual SLA threshold.",
    },
    {
        "id": "KB-005",
        "title": "Incident Escalation SLA",
        "text": "Critical severity security incidents must be escalated to the Compliance Officer within 4 hours of detection.",
    },
    {
        "id": "KB-006",
        "title": "Budget Variance Governance",
        "text": "Unbudgeted infrastructure variances exceeding $25,000 require written approval from the Chief Financial Officer.",
    },
]


@dataclass
class DemoScenario:
    """Single generated document scenario with ground-truth text and expected verdicts."""

    scenario_id: str
    title: str
    description: str
    image_bytes: bytes
    ground_truth_text: str
    expected_claims: list[tuple[str, ClaimVerdict]]
    expected_tamper: bool
    corpus_jsonl: str
    image_path: Path | None = None


@dataclass
class DemoSuite:
    """Collection of all built-in synthetic document forensic scenarios."""

    scenarios: dict[str, DemoScenario]
    corpus_jsonl: str
    corpus_path: Path | None = None


def get_reference_corpus_jsonl() -> str:
    """Return the built-in trusted compliance knowledge base in JSONL format."""
    return "\n".join(json.dumps(item) for item in REFERENCE_PASSAGES) + "\n"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    )
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def _render_memo_image(
    doc_id: str,
    subtitle: str,
    claims: list[str],
    width: int = 1400,
    height: int = 1750,
    dpi: int = 200,
) -> tuple[bytes, list[tuple[int, int, int, int]]]:
    """Render a crisp 200-DPI compliance memorandum PNG and track per-claim bounding boxes."""
    img = Image.new("RGB", (width, height), color=(252, 252, 253))
    draw = ImageDraw.Draw(img)

    font_title = _load_font(38, bold=True)
    font_sub = _load_font(24, bold=False)
    font_meta = _load_font(22, bold=True)
    font_body = _load_font(29, bold=False)
    font_footer = _load_font(20, bold=False)

    # Clean institutional header layout.
    draw.rectangle([(84, 72), (width - 84, 86)], fill=(18, 42, 78))
    draw.text((84, 110), "APEX MERIDIAN SYSTEMS", fill=(18, 28, 46), font=font_title)
    draw.text((84, 162), subtitle, fill=(64, 78, 98), font=font_sub)
    draw.text((84, 206), f"DOCUMENT ID: {doc_id}", fill=(32, 48, 72), font=font_meta)
    draw.line([(84, 248), (width - 84, 248)], fill=(160, 172, 188), width=3)

    y_cursor = 296
    claim_boxes: list[tuple[int, int, int, int]] = []

    for idx, claim in enumerate(claims, start=1):
        wrapped = textwrap.wrap(f"{idx}. {claim}", width=66)
        box_top = max(0, y_cursor - 14)
        for line in wrapped:
            draw.text((92, y_cursor), line, fill=(16, 20, 26), font=font_body)
            y_cursor += 46
        box_bottom = min(height, y_cursor + 10)
        claim_boxes.append((76, box_top, width - 76, box_bottom))
        y_cursor += 34

    draw.line([(84, height - 132), (width - 84, height - 132)], fill=(185, 194, 206), width=2)
    draw.text(
        (84, height - 104),
        "CONFIDENTIAL COMPLIANCE FILING",
        fill=(96, 108, 124),
        font=font_footer,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue(), claim_boxes


def _degrade_skewed_blurry(png_bytes: bytes, seed: int = 2026) -> bytes:
    """Apply geometric skew, Gaussian defocus blur, contrast wash-out, and sensor noise."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]

    # Rotate clockwise by 7.0 deg (negative angle in OpenCV); deskew restores via +7.0 deg.
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, -7.0, 1.0)
    tilted = cv2.warpAffine(
        bgr,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(252, 252, 253),
    )

    # Apply realistic scanner optics defocus blur.
    blurred = cv2.GaussianBlur(tilted, (9, 9), 2.1)

    # Compress dynamic range to simulate faded/washed-out mobile capture + gentle grain.
    washed = np.clip(blurred.astype(np.float32) * 0.34 + 124.0, 0.0, 255.0)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 3.2, size=washed.shape).astype(np.float32)
    noisy = np.clip(washed + noise, 0.0, 255.0).astype(np.uint8)

    rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="PNG", dpi=(200, 200))
    return out.getvalue()


def _inject_tamper_patch(png_bytes: bytes, patch_box: tuple[int, int, int, int], seed: int = 99) -> bytes:
    """Simulate a forged copy-move/re-compressed patch over an altered numeric claim region."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = patch_box
    patch = bgr[y1:y2, x1:x2].copy()

    # Low-quality JPEG recompression (DCT ringing) followed by subtle splice luminance shift & sensor grain.
    ok, jpg_patch = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), 38])
    if not ok:
        raise RuntimeError("Failed to synthesize JPEG splice patch.")
    decoded_patch = cv2.imdecode(jpg_patch, cv2.IMREAD_COLOR)

    rng = np.random.default_rng(seed)
    spliced = np.clip(
        decoded_patch.astype(np.float32) - 9.0 + rng.normal(0.0, 4.5, size=decoded_patch.shape),
        0.0,
        255.0,
    ).astype(np.uint8)
    bgr[y1:y2, x1:x2] = spliced

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="PNG", dpi=(200, 200))
    return out.getvalue()


def generate_demo_suite(output_dir: Path | str | None = None) -> DemoSuite:
    """Generate all 3 realistic synthetic compliance document scenarios and reference JSONL."""
    corpus_jsonl = get_reference_corpus_jsonl()

    clean_claims: list[tuple[str, ClaimVerdict]] = [
        (
            "Apex Meridian Systems reported total Q3 2026 operating settlement expenses of $482,500 across 14 active cloud regions.",
            "SUPPORTED",
        ),
        (
            "The independent SOC 2 Type II compliance audit was completed on October 14, 2026 with zero critical security findings.",
            "SUPPORTED",
        ),
        (
            "All financial ledger backups are encrypted using AES-256-GCM and retained for 7 years under Policy FIN-402.",
            "SUPPORTED",
        ),
        (
            "Primary clearing API availability reached 99.94% during Q3 2026, exceeding the 99.90% contractual SLA threshold.",
            "SUPPORTED",
        ),
        (
            "Critical severity security incidents must be escalated to the Compliance Officer within 4 hours of detection.",
            "SUPPORTED",
        ),
    ]

    tampered_claims: list[tuple[str, ClaimVerdict]] = [
        (
            "Apex Meridian Systems reported total Q3 2026 operating settlement expenses of $940,000 across 14 active cloud regions.",
            "CONTRADICTED",
        ),
        (
            "The independent SOC 2 Type II compliance audit failed on October 14, 2026 with 6 critical security findings.",
            "CONTRADICTED",
        ),
        (
            "All financial ledger backups are encrypted using AES-256-GCM and retained for 7 years under Policy FIN-402.",
            "SUPPORTED",
        ),
        (
            "Primary clearing API availability dropped to 94.10% during Q3 2026, missing the 99.90% contractual SLA threshold.",
            "CONTRADICTED",
        ),
        (
            "The Zurich satellite clearing hub processes 12,000 daily Swiss franc derivatives contracts.",
            "UNVERIFIED",
        ),
    ]

    clean_bytes, _ = _render_memo_image(
        doc_id="AMS-2026-Q3-CLEAN",
        subtitle="Q3 2026 Regulatory Compliance & Settlement Summary",
        claims=[c for c, _ in clean_claims],
    )

    skewed_base_bytes, _ = _render_memo_image(
        doc_id="AMS-2026-Q3-SCAN",
        subtitle="Q3 2026 Regulatory Compliance & Settlement Summary",
        claims=[c for c, _ in clean_claims],
    )
    skewed_bytes = _degrade_skewed_blurry(skewed_base_bytes)

    tampered_base_bytes, tampered_boxes = _render_memo_image(
        doc_id="AMS-2026-Q3-REV",
        subtitle="Q3 2026 Regulatory Compliance & Settlement Summary",
        claims=[c for c, _ in tampered_claims],
    )
    # Inject splice tampering directly onto the altered $940,000 settlement figure region.
    tampered_bytes = _inject_tamper_patch(tampered_base_bytes, tampered_boxes[0])

    scenarios = {
        "clean": DemoScenario(
            scenario_id="clean",
            title="Clean Document",
            description="High-contrast authentic compliance memo whose factual claims match the trusted reference corpus.",
            image_bytes=clean_bytes,
            ground_truth_text="\n".join(c for c, _ in clean_claims),
            expected_claims=clean_claims,
            expected_tamper=False,
            corpus_jsonl=corpus_jsonl,
        ),
        "skewed_blurry": DemoScenario(
            scenario_id="skewed_blurry",
            title="Degraded Scan (Deskew + Restore)",
            description="7-degree tilted, Gaussian-blurred, washed-out scan requiring forensic deskewing and contrast/sharpness recovery.",
            image_bytes=skewed_bytes,
            ground_truth_text="\n".join(c for c, _ in clean_claims),
            expected_claims=clean_claims,
            expected_tamper=False,
            corpus_jsonl=corpus_jsonl,
        ),
        "tampered_hallucinated": DemoScenario(
            scenario_id="tampered_hallucinated",
            title="Tampered / Hallucinated Claim",
            description="Memo containing a spliced low-JPEG-quality patch over altered settlement numbers alongside contradicted and unverified claims.",
            image_bytes=tampered_bytes,
            ground_truth_text="\n".join(c for c, _ in tampered_claims),
            expected_claims=tampered_claims,
            expected_tamper=True,
            corpus_jsonl=corpus_jsonl,
        ),
    }

    corpus_path: Path | None = None
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        corpus_path = out_path / "reference_corpus.jsonl"
        corpus_path.write_text(corpus_jsonl, encoding="utf-8")
        for sid, scenario in scenarios.items():
            img_path = out_path / f"{sid}.png"
            img_path.write_bytes(scenario.image_bytes)
            scenario.image_path = img_path

    return DemoSuite(scenarios=scenarios, corpus_jsonl=corpus_jsonl, corpus_path=corpus_path)
