"""FastAPI endpoint tests and CLI end-to-end smoke tests."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import main as cli_main
from veridoc.server import app
from veridoc.synthetic import generate_demo_suite, get_reference_corpus_jsonl

client = TestClient(app)


def test_ui_and_health_endpoints() -> None:
    ui_res = client.get("/")
    assert ui_res.status_code == 200
    assert "text/html" in ui_res.headers["content-type"]
    assert "VeriDoc Forensics" in ui_res.text

    health_res = client.get("/health")
    assert health_res.status_code == 200
    payload = health_res.json()
    assert payload["status"] == "ok"
    assert payload["tesseract_available"] is True
    assert payload["tesseract_version"]


def test_analyze_demo_scenarios_and_export_html() -> None:
    clean_res = client.post("/api/analyze-demo", json={"scenario": "clean"})
    assert clean_res.status_code == 200
    clean_data = clean_res.json()
    assert clean_data["overall_risk"] == "LOW"
    assert clean_data["forensics"]["tamper_flagged"] is False
    assert clean_data["verification"]["faithfulness_score"] == 1.0

    skew_res = client.post("/api/analyze-demo", json={"scenario": "skewed_blurry"})
    assert skew_res.status_code == 200
    skew_data = skew_res.json()
    assert skew_data["forensics"]["deskew_applied"] is True
    assert skew_data["restored_ocr"]["word_count"] > skew_data["raw_ocr"]["word_count"]
    assert skew_data["verification"]["supported_count"] == 5

    tamper_res = client.post("/api/analyze-demo", json={"scenario": "tampered_hallucinated"})
    assert tamper_res.status_code == 200
    tamper_data = tamper_res.json()
    assert tamper_data["overall_risk"] == "CRITICAL"
    assert tamper_data["forensics"]["tamper_flagged"] is True
    assert tamper_data["verification"]["contradicted_count"] >= 3
    assert tamper_data["verification"]["unverified_count"] >= 1

    # Export standalone HTML report in both dark and light modes.
    export_res = client.post("/api/export-html?theme=light", json=tamper_data)
    assert export_res.status_code == 200
    assert "text/html" in export_res.headers["content-type"]
    assert "veridoc-audit-report.html" in export_res.headers.get("content-disposition", "")
    assert "CRITICAL" in export_res.text
    assert "$940,000" in export_res.text

    missing_res = client.post("/api/analyze-demo", json={"scenario": "does_not_exist"})
    assert missing_res.status_code == 404


def test_verify_multipart_upload_and_direct_claim_text() -> None:
    suite = generate_demo_suite()
    corpus = get_reference_corpus_jsonl()

    # 1. Multipart PNG image upload + JSONL corpus.
    upload_res = client.post(
        "/api/verify",
        files={"file": ("memo.png", suite.scenarios["clean"].image_bytes, "image/png")},
        data={"corpus_jsonl": corpus},
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["forensics"]["width"] == 1400
    assert upload_data["verification"]["supported_count"] == 5

    # 2. Multipart direct claim_text without image.
    text_res = client.post(
        "/api/verify",
        data={
            "claim_text": "Apex Meridian Systems reported total Q3 2026 operating settlement expenses of $940,000 across 14 active cloud regions.",
            "corpus_jsonl": corpus,
        },
    )
    assert text_res.status_code == 200
    text_data = text_res.json()
    assert text_data["forensics"] is None
    assert text_data["verification"]["contradicted_count"] == 1

    # 3. JSON image_base64 verification.
    b64_img = base64.b64encode(suite.scenarios["tampered_hallucinated"].image_bytes).decode("ascii")
    b64_res = client.post(
        "/api/verify",
        json={"image_base64": b64_img, "corpus_jsonl": corpus},
    )
    assert b64_res.status_code == 200
    assert b64_res.json()["forensics"]["tamper_flagged"] is True

    # 4. Invalid payload rejection.
    bad_res = client.post("/api/verify", json={"corpus_jsonl": corpus})
    assert bad_res.status_code == 400


def test_verify_rejects_oversized_request(monkeypatch: object) -> None:
    from veridoc import server

    monkeypatch.setattr(server, "MAX_REQUEST_BODY_BYTES", 64, raising=False)  # type: ignore[attr-defined]
    response = client.post("/api/verify", json={"claim_text": "x" * 128})

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body exceeds the 64-byte limit."


def test_verify_rejects_oversized_stream_without_content_length(monkeypatch: object) -> None:
    from veridoc import server

    monkeypatch.setattr(server, "MAX_REQUEST_BODY_BYTES", 64)
    response = client.post(
        "/api/verify",
        content=b" " * 65,
        headers={"Content-Type": "application/json", "Transfer-Encoding": "chunked"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body exceeds the 64-byte limit."


def test_verify_rejects_malformed_base64_image() -> None:
    response = client.post("/api/verify", json={"image_base64": "%%%not-base64%%%"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image_base64: expected valid base64 data."


def test_verify_rejects_images_over_pixel_limit(monkeypatch: object) -> None:
    from veridoc import forensics

    monkeypatch.setattr(forensics, "MAX_IMAGE_PIXELS", 4)  # type: ignore[attr-defined]
    image = Image.new("RGB", (3, 2), color=(255, 255, 255))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")

    response = client.post(
        "/api/verify",
        files={"file": ("oversized.png", image_bytes.getvalue(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Image dimensions exceed the 4-pixel limit."


def test_cli_demo_verify_image_and_benchmark(tmp_path: Path, capsys: object) -> None:
    report_path = tmp_path / "demo_report.html"
    rc_demo = cli_main(["--demo", "clean", "--export-html", str(report_path)])
    assert rc_demo == 0
    assert report_path.is_file()
    assert "VeriDoc Forensics" in report_path.read_text(encoding="utf-8")

    # Save an image and custom corpus on disk and verify via --verify-image.
    suite = generate_demo_suite(output_dir=tmp_path / "suite")
    img_path = suite.scenarios["tampered_hallucinated"].image_path
    corpus_path = suite.corpus_path
    assert img_path is not None and corpus_path is not None

    rc_img = cli_main(["--verify-image", str(img_path), "--corpus", str(corpus_path), "--json"])
    assert rc_img == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    parsed_json = json.loads(captured.out[captured.out.index("{") :])
    assert parsed_json["forensics"]["tamper_flagged"] is True
    assert parsed_json["verification"]["contradicted_count"] >= 3

    # Full benchmark via CLI --benchmark --json.
    rc_bench = cli_main(["--benchmark", "--json"])
    assert rc_bench == 0
    bench_out = capsys.readouterr()  # type: ignore[attr-defined]
    bench_data = json.loads(bench_out.out)
    assert bench_data["mean_restored_char_acc"] > bench_data["mean_raw_char_acc"]
    assert bench_data["overall_claim_accuracy"] >= 0.95
    assert bench_data["tamper_detection_accuracy"] == 1.0


def test_studio_ui_disables_heuristic_caching() -> None:
    """The studio UI sends no cache validators, so browsers heuristically cached
    stale HTML after edits (observed live in the stdlib-served fleet)."""
    resp = client.get("/")
    assert resp.headers.get("cache-control") == "no-cache"
