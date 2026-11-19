"""Unit and integration tests for OpenCV/Pillow image forensics and restoration."""

from __future__ import annotations

import io
import struct
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image

from veridoc.forensics import DocumentForensicAnalyzer, build_before_after_svg, encode_image_base64
from veridoc.synthetic import generate_demo_suite


def test_clean_document_passes_forensics() -> None:
    suite = generate_demo_suite()
    analyzer = DocumentForensicAnalyzer()
    restored_bytes, report = analyzer.analyze_and_restore(suite.scenarios["clean"].image_bytes)

    assert len(restored_bytes) > 0
    assert report.width == 1400
    assert report.height == 1750
    assert report.dpi_adequate is True
    assert report.is_blurry is False
    assert report.needs_deskew is False
    assert report.contrast_adequate is True
    assert report.tamper_flagged is False
    assert report.tamper_suspicion_score < 0.52
    assert report.quality_score >= 90.0
    assert report.restoration_applied == []
    assert report.raw_image_base64 and report.raw_image_base64.startswith("data:image/png;base64,")
    assert report.comparison_svg and "<svg" in report.comparison_svg


def test_skewed_blurry_document_is_detected_and_restored() -> None:
    suite = generate_demo_suite()
    analyzer = DocumentForensicAnalyzer()
    skewed_bytes = suite.scenarios["skewed_blurry"].image_bytes

    restored_bytes, report = analyzer.analyze_and_restore(skewed_bytes, include_previews=False)

    assert report.is_blurry is True
    assert report.needs_deskew is True
    assert report.deskew_applied is True
    assert abs(abs(report.skew_angle_deg) - 7.0) <= 1.0
    assert report.contrast_adequate is False
    assert report.tamper_flagged is False
    assert "deskew" in " ".join(report.restoration_applied)
    assert "clahe_contrast_stretch" in report.restoration_applied
    assert "denoise_unsharp_mask" in report.restoration_applied

    # Re-auditing the restored output must verify skew reduction and sharpness gain.
    _, post_report = analyzer.analyze_and_restore(restored_bytes, include_previews=False)
    assert abs(post_report.skew_angle_deg) <= 1.0
    assert post_report.blur_laplacian_var > report.blur_laplacian_var * 2.0
    assert post_report.dynamic_range > report.dynamic_range


def test_tampered_document_triggers_ela_and_noise_anomaly_alerts() -> None:
    suite = generate_demo_suite()
    analyzer = DocumentForensicAnalyzer()
    tampered_bytes = suite.scenarios["tampered_hallucinated"].image_bytes

    _, report = analyzer.analyze_and_restore(tampered_bytes, include_previews=True)

    assert report.tamper_flagged is True
    assert report.tamper_suspicion_score >= 0.52
    assert len(report.suspicious_regions) >= 1
    assert any(issue.code == "TAMPER_SUSPECTED" and issue.severity == "critical" for issue in report.issues)
    assert report.ela_heatmap_base64 is not None
    assert report.comparison_svg is not None and "TAMPER ALERT" in report.comparison_svg


def test_custom_numpy_and_pil_inputs_plus_svg_helper() -> None:
    img = Image.new("RGB", (640, 480), color=(250, 250, 250))
    arr = np.array(img)
    cv2.putText(arr, "Audit Test 1234", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (15, 15, 15), 2)

    analyzer = DocumentForensicAnalyzer()
    restored_from_np, rep_np = analyzer.analyze_and_restore(arr, include_previews=False)
    restored_from_pil, rep_pil = analyzer.analyze_and_restore(Image.fromarray(arr), include_previews=False)

    assert rep_np.width == 640 and rep_pil.height == 480
    svg = build_before_after_svg(restored_from_np, restored_from_pil, rep_np)
    assert svg.startswith("<svg") and "Forensic-Restored Output" in svg
    assert encode_image_base64(b"abc").startswith("data:image/png;base64,")


def test_forensic_analyzer_rejects_oversized_image_before_decode() -> None:
    image = Image.new("RGB", (1, 1), color=(255, 255, 255))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png = bytearray(buf.getvalue())

    # Keep a valid PNG header but declare dimensions above the analyzer limit.
    struct.pack_into(">II", png, 16, 5_000, 5_000)
    png[29:33] = struct.pack(">I", zlib.crc32(png[12:29]) & 0xFFFFFFFF)

    with pytest.raises(ValueError, match="pixel limit"):
        DocumentForensicAnalyzer().analyze_and_restore(bytes(png), include_previews=False)


def test_forensic_analyzer_rejects_invalid_inputs() -> None:
    analyzer = DocumentForensicAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyze_and_restore(b"")
    with pytest.raises(ValueError):
        analyzer.analyze_and_restore(b"not-a-valid-image")
    with pytest.raises(TypeError):
        analyzer.analyze_and_restore(12345)  # type: ignore[arg-type]

    # Blank tiny image should still run without crashing.
    blank = Image.new("RGB", (128, 128), color=(255, 255, 255))
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    _, blank_report = analyzer.analyze_and_restore(buf.getvalue(), include_previews=False)
    assert blank_report.dpi_adequate is False
    assert blank_report.skew_angle_deg == 0.0
