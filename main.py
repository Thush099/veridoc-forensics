"""CLI entrypoint for VeriDoc Forensics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import uvicorn

from veridoc.benchmark import format_benchmark_table, run_benchmark
from veridoc.models import AuditResult
from veridoc.pipeline import VeriDocPipeline, render_audit_html
from veridoc.synthetic import get_reference_corpus_jsonl


def _print_audit_summary(audit: AuditResult) -> None:
    print(f"\n=== Scenario: {audit.scenario_id or 'custom'} [{audit.overall_risk} RISK] ===")
    print(audit.summary_banner)
    if audit.forensics:
        f = audit.forensics
        restorations = ", ".join(f.restoration_applied) if f.restoration_applied else "none"
        print(
            f"Forensics : quality={f.quality_score:.1f}/100 | blur={f.blur_laplacian_var:.1f} | "
            f"skew={f.skew_angle_deg:+.2f}deg | dyn_range={f.dynamic_range:.1f} | "
            f"tamper={f.tamper_suspicion_score:.3f} ({'FLAGGED' if f.tamper_flagged else 'CLEAN'}) | "
            f"restorations=[{restorations}]"
        )
    if audit.raw_ocr and audit.raw_ocr is not audit.restored_ocr:
        print(
            f"Raw OCR   : {audit.raw_ocr.word_count} words | mean_conf={audit.raw_ocr.mean_confidence:.1f}%"
        )
    print(
        f"Rest. OCR : {audit.restored_ocr.word_count} words | "
        f"mean_conf={audit.restored_ocr.mean_confidence:.1f}% | engine={audit.restored_ocr.engine}"
    )
    v = audit.verification
    print(
        f"Verdicts  : faithfulness={v.faithfulness_score:.1%} | "
        f"supported={v.supported_count}, partial={v.partial_count}, "
        f"contradicted={v.contradicted_count}, unverified={v.unverified_count}"
    )
    for claim in v.claims:
        ev_id = claim.matched_passages[0].passage_id if claim.matched_passages else "NONE"
        print(f"  - [{claim.verdict:<12}] ({ev_id}) {claim.claim_text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veridoc",
        description="VeriDoc Forensics — Multimodal Document OCR, Image Quality Auditor & Claim Verifier",
    )
    parser.add_argument(
        "--demo",
        nargs="?",
        const="all",
        choices=["clean", "skewed_blurry", "tampered_hallucinated", "all"],
        help="Run built-in synthetic document scenario(s).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run raw vs restored OCR and claim verification benchmark suite.",
    )
    parser.add_argument(
        "--verify-image",
        type=Path,
        help="Path to a PNG/JPG document image to audit.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Optional path to trusted knowledge-base JSONL file (defaults to built-in compliance corpus).",
    )
    parser.add_argument(
        "--export-html",
        type=Path,
        help="Write a standalone HTML forensic audit report to this path.",
    )
    parser.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="HTML export report theme (default: dark).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI Studio web server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for --serve mode (default: 8000).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output for --demo, --benchmark, or --verify-image.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.serve:
        uvicorn.run("veridoc.server:app", host="0.0.0.0", port=args.port, reload=False)
        return 0

    if args.benchmark:
        bench = run_benchmark()
        if args.json:
            print(json.dumps(bench.to_dict(), indent=2))
        else:
            print(format_benchmark_table(bench))
        return 0

    pipeline = VeriDocPipeline()

    if args.verify_image:
        if not args.verify_image.is_file():
            print(f"Error: image file not found: {args.verify_image}", file=sys.stderr)
            return 2
        corpus_text = (
            args.corpus.read_text(encoding="utf-8") if args.corpus else get_reference_corpus_jsonl()
        )
        audit = pipeline.audit_image(
            image_input=args.verify_image,
            corpus=corpus_text,
            scenario_id=args.verify_image.stem,
            include_raw_ocr=True,
            include_previews=bool(args.export_html),
        )
        if args.export_html:
            args.export_html.parent.mkdir(parents=True, exist_ok=True)
            args.export_html.write_text(render_audit_html(audit, theme=args.theme), encoding="utf-8")
            print(f"Exported standalone HTML report to {args.export_html}")
        if args.json:
            print(audit.model_dump_json(indent=2))
        else:
            _print_audit_summary(audit)
        return 0

    if args.demo:
        targets = (
            ["clean", "skewed_blurry", "tampered_hallucinated"]
            if args.demo == "all"
            else [args.demo]
        )
        audits = [
            pipeline.audit_demo(sid, include_previews=bool(args.export_html)) for sid in targets
        ]
        if args.export_html:
            # Export the selected scenario (or tampered_hallucinated showcase when running all).
            chosen = audits[-1]
            args.export_html.parent.mkdir(parents=True, exist_ok=True)
            args.export_html.write_text(render_audit_html(chosen, theme=args.theme), encoding="utf-8")
            print(f"Exported standalone HTML report to {args.export_html}")
        if args.json:
            payload = [a.model_dump(mode="json") for a in audits] if len(audits) > 1 else audits[0].model_dump(mode="json")
            print(json.dumps(payload, indent=2))
        else:
            for audit in audits:
                _print_audit_summary(audit)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
