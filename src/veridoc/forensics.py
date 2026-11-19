"""Computer-vision document quality forensics, deskewing, restoration, and ELA tamper auditing."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from veridoc.models import ForensicIssue, ForensicReport, SuspiciousRegion

ImageInput = Union[bytes, str, Path, np.ndarray, Image.Image]
MAX_IMAGE_PIXELS = 20_000_000


class _ImageTooLargeError(ValueError):
    pass


def encode_image_base64(image_bytes: bytes, mime: str = "image/png") -> str:
    """Encode raw image bytes as a data URI suitable for JSON, HTML, or SVG embedding."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_before_after_svg(
    raw_bytes: bytes,
    restored_bytes: bytes,
    report: ForensicReport,
    width: int = 920,
    height: int = 560,
) -> str:
    """Render a standalone SVG comparing the raw input scan and forensic-restored document."""
    raw_uri = encode_image_base64(raw_bytes, "image/png")
    restored_uri = encode_image_base64(restored_bytes, "image/png")

    panel_w = 420
    panel_h = 460
    left_x = 24
    right_x = 476
    panel_y = 66

    # Map suspicious regions from image coordinates into the left preview panel.
    img_w = max(report.width, 1)
    img_h = max(report.height, 1)
    scale = min(panel_w / img_w, panel_h / img_h)
    rendered_w = img_w * scale
    rendered_h = img_h * scale
    offset_x = left_x + (panel_w - rendered_w) / 2.0
    offset_y = panel_y + (panel_h - rendered_h) / 2.0

    overlay_rects = []
    for region in report.suspicious_regions[:6]:
        rx = offset_x + region.x * scale
        ry = offset_y + region.y * scale
        rw = max(region.width * scale, 10.0)
        rh = max(region.height * scale, 10.0)
        overlay_rects.append(
            f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
            f'fill="rgba(244,63,94,0.18)" stroke="#f43f5e" stroke-width="2" stroke-dasharray="5,3"/>'
        )

    tamper_color = "#f43f5e" if report.tamper_flagged else "#10b981"
    tamper_label = (
        f"TAMPER ALERT ({report.tamper_suspicion_score:.2f})"
        if report.tamper_flagged
        else f"AUTHENTIC ({report.tamper_suspicion_score:.2f})"
    )
    restorations = ", ".join(report.restoration_applied) if report.restoration_applied else "none needed"
    summary_text = html.escape(
        f"Quality {report.quality_score:.1f}/100 | Skew {report.skew_angle_deg:+.2f} deg | "
        f"Blur Var {report.blur_laplacian_var:.1f} | Restored: {restorations}"
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="Before and after forensic comparison">'
        f'<rect width="{width}" height="{height}" rx="12" fill="#0b0f17" stroke="#1f2a3d"/>'
        f'<text x="24" y="30" fill="#e2e8f0" font-family="ui-sans-serif,system-ui,sans-serif" '
        f'font-size="16" font-weight="700">VeriDoc Forensic Comparison</text>'
        f'<text x="24" y="50" fill="#94a3b8" font-family="ui-monospace,monospace" '
        f'font-size="11">{summary_text}</text>'
        f'<rect x="{width - 210}" y="16" width="186" height="26" rx="13" fill="#131a26" stroke="{tamper_color}"/>'
        f'<text x="{width - 117}" y="33" text-anchor="middle" fill="{tamper_color}" '
        f'font-family="ui-monospace,monospace" font-size="11" font-weight="700">{html.escape(tamper_label)}</text>'
        f'<rect x="{left_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="8" fill="#111827" stroke="#263248"/>'
        f'<image href="{raw_uri}" x="{left_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" preserveAspectRatio="xMidYMid meet"/>'
        f'{"".join(overlay_rects)}'
        f'<text x="{left_x + panel_w / 2}" y="{panel_y + panel_h + 20}" text-anchor="middle" '
        f'fill="#94a3b8" font-family="ui-sans-serif,system-ui,sans-serif" font-size="12">Raw Input Scan</text>'
        f'<rect x="{right_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="8" fill="#111827" stroke="#263248"/>'
        f'<image href="{restored_uri}" x="{right_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" preserveAspectRatio="xMidYMid meet"/>'
        f'<text x="{right_x + panel_w / 2}" y="{panel_y + panel_h + 20}" text-anchor="middle" '
        f'fill="#38bdf8" font-family="ui-sans-serif,system-ui,sans-serif" font-size="12">Forensic-Restored Output</text>'
        f"</svg>"
    )


class DocumentForensicAnalyzer:
    """Inspects scanned document images for quality degradation and splice tampering."""

    def __init__(
        self,
        blur_threshold: float = 180.0,
        skew_threshold_deg: float = 0.8,
        min_dpi: float = 135.0,
        tamper_threshold: float = 0.52,
        tile_size: int = 64,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.skew_threshold_deg = skew_threshold_deg
        self.min_dpi = min_dpi
        self.tamper_threshold = tamper_threshold
        self.tile_size = tile_size

    def analyze_and_restore(
        self,
        image_input: ImageInput,
        include_previews: bool = True,
    ) -> tuple[bytes, ForensicReport]:
        """Run full forensic audit and return ``(restored_png_bytes, ForensicReport)``."""
        bgr, raw_png_bytes, meta_dpi = self._load_image(image_input)
        height, width = bgr.shape[:2]
        channels = 1 if bgr.ndim == 2 else bgr.shape[2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if channels >= 3 else bgr.copy()

        estimated_dpi = self._estimate_dpi(width, height, meta_dpi)
        dpi_adequate = estimated_dpi >= self.min_dpi and min(width, height) >= 320

        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = blur_var < self.blur_threshold

        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))
        p1, p99 = np.percentile(gray, [1, 99])
        dynamic_range = float(p99 - p1)
        highlight_clip_ratio = float(np.mean(gray >= 253))
        shadow_clip_ratio = float(np.mean(gray <= 2))
        contrast_adequate = (
            dynamic_range >= 100.0
            and contrast_std >= 18.0
            and 45.0 <= mean_brightness <= 253.5
        )

        # Estimate geometric skew via horizontal projection-profile variance maximization.
        skew_angle_deg = self._estimate_skew_angle(gray)
        needs_deskew = abs(skew_angle_deg) >= self.skew_threshold_deg

        # Error Level Analysis + local smooth-tile noise inconsistency audit.
        (
            ela_mean,
            ela_max_tile,
            ela_tile_var,
            noise_anomaly_score,
            tamper_score,
            suspicious_regions,
            ela_heatmap_bytes,
        ) = self._run_tamper_forensics(bgr, gray)
        tamper_flagged = tamper_score >= self.tamper_threshold

        # Apply non-destructive geometric & photometric restoration only when needed.
        restored_bgr, restoration_applied, deskew_applied = self._restore_image(
            bgr=bgr,
            gray=gray,
            skew_angle_deg=skew_angle_deg,
            needs_deskew=needs_deskew,
            is_blurry=is_blurry,
            contrast_adequate=contrast_adequate,
            dynamic_range=dynamic_range,
        )
        restored_png_bytes = self._encode_png(restored_bgr, dpi=int(round(max(estimated_dpi, 150.0))))

        issues = self._collect_issues(
            dpi_adequate=dpi_adequate,
            estimated_dpi=estimated_dpi,
            is_blurry=is_blurry,
            blur_var=blur_var,
            needs_deskew=needs_deskew,
            skew_angle_deg=skew_angle_deg,
            contrast_adequate=contrast_adequate,
            dynamic_range=dynamic_range,
            tamper_flagged=tamper_flagged,
            tamper_score=tamper_score,
        )

        quality_score = self._compute_quality_score(
            dpi_adequate=dpi_adequate,
            blur_var=blur_var,
            skew_angle_deg=skew_angle_deg,
            dynamic_range=dynamic_range,
            contrast_std=contrast_std,
            tamper_score=tamper_score,
        )

        report = ForensicReport(
            width=width,
            height=height,
            channels=channels,
            estimated_dpi=round(estimated_dpi, 1),
            dpi_adequate=dpi_adequate,
            blur_laplacian_var=round(blur_var, 2),
            is_blurry=is_blurry,
            skew_angle_deg=round(skew_angle_deg, 2),
            needs_deskew=needs_deskew,
            deskew_applied=deskew_applied,
            mean_brightness=round(mean_brightness, 2),
            contrast_std=round(contrast_std, 2),
            dynamic_range=round(dynamic_range, 2),
            highlight_clip_ratio=round(highlight_clip_ratio, 4),
            shadow_clip_ratio=round(shadow_clip_ratio, 4),
            contrast_adequate=contrast_adequate,
            ela_mean_residual=round(ela_mean, 3),
            ela_max_tile_residual=round(ela_max_tile, 3),
            ela_tile_variance=round(ela_tile_var, 3),
            local_noise_anomaly_score=round(noise_anomaly_score, 3),
            tamper_suspicion_score=round(tamper_score, 3),
            tamper_flagged=tamper_flagged,
            suspicious_regions=suspicious_regions,
            quality_score=round(quality_score, 1),
            restoration_applied=restoration_applied,
            issues=issues,
        )

        if include_previews:
            report.raw_image_base64 = encode_image_base64(raw_png_bytes, "image/png")
            report.restored_image_base64 = encode_image_base64(restored_png_bytes, "image/png")
            report.ela_heatmap_base64 = encode_image_base64(ela_heatmap_bytes, "image/png")
            report.comparison_svg = build_before_after_svg(raw_png_bytes, restored_png_bytes, report)

        return restored_png_bytes, report

    def _load_image(self, image_input: ImageInput) -> tuple[np.ndarray, bytes, float | None]:
        meta_dpi: float | None = None

        if isinstance(image_input, (str, Path)):
            raw_bytes = Path(image_input).read_bytes()
            return self._decode_bytes(raw_bytes)

        if isinstance(image_input, bytes):
            return self._decode_bytes(image_input)

        if isinstance(image_input, Image.Image):
            self._validate_image_size(*image_input.size)
            dpi_info = image_input.info.get("dpi")
            if isinstance(dpi_info, (tuple, list)) and dpi_info and dpi_info[0]:
                meta_dpi = float(dpi_info[0])
            rgb = image_input.convert("RGB")
            arr = np.array(rgb)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return bgr, self._encode_png(bgr), meta_dpi

        if isinstance(image_input, np.ndarray):
            if image_input.ndim >= 2:
                self._validate_image_size(image_input.shape[1], image_input.shape[0])
            arr = image_input.copy()
            if arr.ndim == 2:
                bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.shape[2] == 4:
                bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            else:
                bgr = arr
            return bgr, self._encode_png(bgr), None

        raise TypeError(f"Unsupported image input type: {type(image_input).__name__}")

    @staticmethod
    def _validate_image_size(width: int, height: int) -> None:
        if width * height > MAX_IMAGE_PIXELS:
            raise _ImageTooLargeError(
                f"Image dimensions exceed the {MAX_IMAGE_PIXELS}-pixel limit."
            )

    def _decode_bytes(self, raw_bytes: bytes) -> tuple[np.ndarray, bytes, float | None]:
        if not raw_bytes:
            raise ValueError("Empty image payload.")
        meta_dpi: float | None = None
        try:
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                self._validate_image_size(*pil_img.size)
                dpi_info = pil_img.info.get("dpi")
                if isinstance(dpi_info, (tuple, list)) and dpi_info and dpi_info[0]:
                    meta_dpi = float(dpi_info[0])
                rgb = pil_img.convert("RGB")
                bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        except _ImageTooLargeError:
            raise
        except Image.DecompressionBombError as exc:
            raise ValueError(
                f"Image dimensions exceed the {MAX_IMAGE_PIXELS}-pixel limit."
            ) from exc
        except Exception as exc:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Unable to decode image bytes: {exc}") from exc
            self._validate_image_size(bgr.shape[1], bgr.shape[0])

        png_bytes = self._encode_png(bgr)
        return bgr, png_bytes, meta_dpi

    @staticmethod
    def _encode_png(bgr: np.ndarray, dpi: int | None = None) -> bytes:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        save_kwargs = {"format": "PNG"}
        if dpi and dpi > 0:
            save_kwargs["dpi"] = (dpi, dpi)
        pil_img.save(buf, **save_kwargs)
        return buf.getvalue()

    @staticmethod
    def _estimate_dpi(width: int, height: int, meta_dpi: float | None) -> float:
        # Estimate effective resolution assuming a standard ~8.5 in letter document width.
        pixel_dpi = width / 8.5
        if meta_dpi and meta_dpi > 20.0:
            if width < 600 or height < 400:
                return min(meta_dpi, max(pixel_dpi, width / 5.0))
            return meta_dpi
        return pixel_dpi

    @staticmethod
    def _estimate_skew_angle(gray: np.ndarray) -> float:
        """Estimate page skew in degrees via horizontal projection-profile variance.

        A positive returned angle means the document should be rotated counter-clockwise
        by ``+skew_angle_deg`` (or clockwise by ``-skew_angle_deg``)?
        To match standard tilt intuition: ``skew_angle_deg > 0`` represents a clockwise
        tilt of the scan, so deskewing applies OpenCV CCW rotation ``+skew_angle_deg``.
        """
        h, w = gray.shape[:2]
        if h < 40 or w < 40:
            return 0.0

        # Downscale for fast, clean search across [-15, +15] degrees.
        target_w = min(640, w)
        scale = target_w / float(w)
        small = cv2.resize(gray, (target_w, max(40, int(h * scale))), interpolation=cv2.INTER_AREA)

        # Normalize local illumination so washed-out scans binarize cleanly.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = clahe.apply(small)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Ignore outer 4% border where tilted scan corners can sit.
        sh, sw = binary.shape
        margin_y = max(4, int(sh * 0.04))
        margin_x = max(4, int(sw * 0.04))
        roi = binary[margin_y : sh - margin_y, margin_x : sw - margin_x]
        if np.count_nonzero(roi) < 60:
            return 0.0

        # Gently bridge horizontal character gaps so text lines form solid horizontal bands.
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        banded = cv2.dilate(roi, horiz_kernel, iterations=1)
        rh, rw = banded.shape
        center = (rw / 2.0, rh / 2.0)

        def projection_score(ccw_angle: float) -> float:
            mat = cv2.getRotationMatrix2D(center, ccw_angle, 1.0)
            rot = cv2.warpAffine(banded, mat, (rw, rh), flags=cv2.INTER_NEAREST, borderValue=0)
            row_sums = rot.sum(axis=1, dtype=np.float64)
            return float(np.var(row_sums))

        best_angle = 0.0
        best_score = -1.0
        for angle in np.linspace(-15.0, 15.0, 61):
            score = projection_score(float(angle))
            if score > best_score:
                best_score = score
                best_angle = float(angle)

        # Fine refinement around coarse peak at 0.1 degree steps.
        fine_start = max(-15.0, best_angle - 0.8)
        fine_end = min(15.0, best_angle + 0.8)
        for angle in np.linspace(fine_start, fine_end, 17):
            score = projection_score(float(angle))
            if score > best_score:
                best_score = score
                best_angle = float(angle)

        # Near-zero ties default to 0.0 to avoid needless resampling.
        zero_score = projection_score(0.0)
        if abs(best_angle) < 0.8 or best_score <= zero_score * 1.015:
            return 0.0
        return best_angle

    def _run_tamper_forensics(
        self,
        bgr: np.ndarray,
        gray: np.ndarray,
    ) -> tuple[float, float, float, float, float, list[SuspiciousRegion], bytes]:
        """Compute ELA residual tiles and local smooth-background noise inconsistencies."""
        ok, jpg_buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Failed to run in-memory JPEG re-compression for ELA.")
        recomp = cv2.imdecode(jpg_buf, cv2.IMREAD_COLOR)
        ela_diff = np.abs(bgr.astype(np.float32) - recomp.astype(np.float32))
        ela_map = np.mean(ela_diff, axis=2)

        gray_f = gray.astype(np.float32)
        smooth_ref = cv2.GaussianBlur(gray_f, (5, 5), 1.0)
        gx = cv2.Sobel(smooth_ref, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth_ref, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy)

        # Dilate strong edges so smooth-paper mask never includes glyph anti-aliasing halos.
        edge_dilated = cv2.dilate((grad_mag > 18.0).astype(np.uint8), np.ones((7, 7), np.uint8))
        smooth_mask = edge_dilated == 0
        hf_residual = np.abs(gray_f - smooth_ref)

        h, w = gray.shape
        ts = max(32, min(self.tile_size, min(h, w) // 2)) if min(h, w) >= 64 else max(16, min(h, w))

        tile_smooth_ela: list[float] = []
        tile_noise_stds: list[float] = []
        tile_bg_meds: list[float] = []
        tile_raw_ela: list[float] = []
        tile_records: list[dict[str, float | int]] = []

        for y in range(0, h - ts + 1, ts):
            for x in range(0, w - ts + 1, ts):
                ela_tile = ela_map[y : y + ts, x : x + ts]
                hf_tile = hf_residual[y : y + ts, x : x + ts]
                gray_tile = gray_f[y : y + ts, x : x + ts]
                sm_tile = smooth_mask[y : y + ts, x : x + ts]

                mean_ela = float(np.mean(ela_tile))
                if int(np.count_nonzero(sm_tile)) >= max(64, (ts * ts) // 8):
                    smooth_ela = float(np.mean(ela_tile[sm_tile]))
                    noise_std = float(np.std(hf_tile[sm_tile]))
                    bg_med = float(np.median(gray_tile[sm_tile]))
                else:
                    continue

                tile_raw_ela.append(mean_ela)
                tile_smooth_ela.append(smooth_ela)
                tile_noise_stds.append(noise_std)
                tile_bg_meds.append(bg_med)
                tile_records.append(
                    {
                        "x": x,
                        "y": y,
                        "width": ts,
                        "height": ts,
                        "mean_ela": mean_ela,
                        "smooth_ela": smooth_ela,
                        "noise_std": noise_std,
                        "bg_med": bg_med,
                    }
                )

        if not tile_records:
            heatmap_png = self._render_ela_heatmap(ela_map, [])
            return 0.0, 0.0, 0.0, 0.0, 0.0, [], heatmap_png

        ela_arr = np.asarray(tile_smooth_ela, dtype=np.float64)
        noise_arr = np.asarray(tile_noise_stds, dtype=np.float64)
        bg_arr = np.asarray(tile_bg_meds, dtype=np.float64)
        raw_ela_arr = np.asarray(tile_raw_ela, dtype=np.float64)

        ela_med = float(np.median(ela_arr))
        ela_mad = float(np.median(np.abs(ela_arr - ela_med)))
        raw_ela_z = np.maximum(0.0, 0.6745 * (ela_arr - ela_med) / (ela_mad + 0.14))
        ela_zscores = np.where(ela_arr >= 0.90, raw_ela_z, 0.0)

        noise_med = float(np.median(noise_arr))
        noise_mad = float(np.median(np.abs(noise_arr - noise_med)))
        raw_noise_z = np.maximum(0.0, 0.6745 * (noise_arr - noise_med) / (noise_mad + 0.08))
        noise_zscores = np.where(noise_arr >= 0.40, raw_noise_z, 0.0)

        bg_med_all = float(np.median(bg_arr))
        bg_mad_all = float(np.median(np.abs(bg_arr - bg_med_all)))
        raw_bg_z = np.maximum(0.0, 0.6745 * np.abs(bg_arr - bg_med_all) / (bg_mad_all + 2.5))
        bg_zscores = np.where(np.abs(bg_arr - bg_med_all) >= 4.0, raw_bg_z, 0.0)

        ela_mean_residual = float(np.mean(ela_map))
        ela_max_tile_residual = float(np.max(raw_ela_arr))
        ela_tile_variance = float(np.var(ela_arr))

        max_noise_z = float(np.max(noise_zscores))
        local_noise_anomaly_score = float(np.clip((max_noise_z - 1.5) / 6.0, 0.0, 1.0))

        combined_tile_z = 0.45 * noise_zscores + 0.35 * ela_zscores + 0.20 * bg_zscores
        peak_z = float(np.max(combined_tile_z))
        tamper_score = float(np.clip((peak_z - 2.2) / 6.0, 0.0, 1.0))

        suspicious: list[SuspiciousRegion] = []
        ranked_indices = np.argsort(-combined_tile_z)
        for idx in ranked_indices:
            if combined_tile_z[idx] < 3.0:
                break
            rec = tile_records[int(idx)]
            suspicious.append(
                SuspiciousRegion(
                    x=int(rec["x"]),
                    y=int(rec["y"]),
                    width=int(rec["width"]),
                    height=int(rec["height"]),
                    ela_zscore=round(float(ela_zscores[idx]), 2),
                    noise_anomaly=round(float(noise_zscores[idx]), 2),
                )
            )
            if len(suspicious) >= 8:
                break

        heatmap_png = self._render_ela_heatmap(ela_map, suspicious if tamper_score >= self.tamper_threshold else [])
        return (
            ela_mean_residual,
            ela_max_tile_residual,
            ela_tile_variance,
            local_noise_anomaly_score,
            tamper_score,
            suspicious,
            heatmap_png,
        )

    @staticmethod
    def _render_ela_heatmap(ela_map: np.ndarray, suspicious: list[SuspiciousRegion]) -> bytes:
        scaled = np.clip(ela_map * 14.0, 0.0, 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
        for region in suspicious:
            cv2.rectangle(
                heat,
                (region.x, region.y),
                (region.x + region.width, region.y + region.height),
                (0, 255, 255),
                2,
            )
        ok, buf = cv2.imencode(".png", heat)
        if not ok:
            raise RuntimeError("Failed to encode ELA heatmap PNG.")
        return buf.tobytes()

    @staticmethod
    def _restore_image(
        bgr: np.ndarray,
        gray: np.ndarray,
        skew_angle_deg: float,
        needs_deskew: bool,
        is_blurry: bool,
        contrast_adequate: bool,
        dynamic_range: float,
    ) -> tuple[np.ndarray, list[str], bool]:
        restored = bgr.copy()
        applied: list[str] = []
        deskew_applied = False

        if needs_deskew:
            h, w = restored.shape[:2]
            # Sample median paper color near top/bottom borders so deskew edges match paper tone.
            border_samples = np.concatenate([gray[0:10, :].ravel(), gray[-10:, :].ravel()])
            bg_val = int(np.median(border_samples)) if border_samples.size else 255
            mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), skew_angle_deg, 1.0)
            restored = cv2.warpAffine(
                restored,
                mat,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(bg_val, bg_val, bg_val),
            )
            deskew_applied = True
            applied.append(f"deskew({skew_angle_deg:+.2f}deg)")

        if not contrast_adequate or dynamic_range < 105.0:
            lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l_chan)
            # Gentle percentile stretch anchors paper background cleanly to white and ink to dark.
            p1, p99 = np.percentile(l_clahe, [1.5, 97.5])
            if p99 - p1 > 10.0:
                l_stretch = np.clip((l_clahe.astype(np.float32) - p1) * (255.0 / (p99 - p1)), 0, 255).astype(np.uint8)
            else:
                l_stretch = l_clahe
            restored = cv2.cvtColor(cv2.merge([l_stretch, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
            applied.append("clahe_contrast_stretch")

        if is_blurry:
            # Gentle bilateral filtering suppresses background grain while preserving stroke edges.
            denoised = cv2.bilateralFilter(restored, d=5, sigmaColor=20, sigmaSpace=20)
            blurred = cv2.GaussianBlur(denoised, (0, 0), 1.4)
            restored = cv2.addWeighted(denoised, 1.75, blurred, -0.75, 0)
            applied.append("denoise_unsharp_mask")

        return restored, applied, deskew_applied

    @staticmethod
    def _collect_issues(
        dpi_adequate: bool,
        estimated_dpi: float,
        is_blurry: bool,
        blur_var: float,
        needs_deskew: bool,
        skew_angle_deg: float,
        contrast_adequate: bool,
        dynamic_range: float,
        tamper_flagged: bool,
        tamper_score: float,
    ) -> list[ForensicIssue]:
        issues: list[ForensicIssue] = []
        if not dpi_adequate:
            issues.append(
                ForensicIssue(
                    code="LOW_DPI",
                    severity="warning",
                    message=f"Low effective resolution ({estimated_dpi:.1f} DPI); OCR glyph fidelity may degrade.",
                    metric_value=round(estimated_dpi, 1),
                )
            )
        if is_blurry:
            issues.append(
                ForensicIssue(
                    code="BLUR_DETECTED",
                    severity="warning",
                    message=f"Low Laplacian sharpness variance ({blur_var:.1f}); unsharp restoration triggered.",
                    metric_value=round(blur_var, 2),
                )
            )
        if needs_deskew:
            issues.append(
                ForensicIssue(
                    code="SKEW_DETECTED",
                    severity="warning",
                    message=f"Page tilt of {skew_angle_deg:+.2f} deg detected; automatic deskew applied.",
                    metric_value=round(skew_angle_deg, 2),
                )
            )
        if not contrast_adequate:
            issues.append(
                ForensicIssue(
                    code="LOW_CONTRAST",
                    severity="warning",
                    message=f"Compressed dynamic range ({dynamic_range:.1f}); CLAHE contrast normalization applied.",
                    metric_value=round(dynamic_range, 2),
                )
            )
        if tamper_flagged:
            issues.append(
                ForensicIssue(
                    code="TAMPER_SUSPECTED",
                    severity="critical",
                    message=f"Anomalous ELA/noise patch detected (tamper suspicion {tamper_score:.2f}); manual review required.",
                    metric_value=round(tamper_score, 3),
                )
            )
        if not issues:
            issues.append(
                ForensicIssue(
                    code="QUALITY_OK",
                    severity="info",
                    message="Document meets sharpness, alignment, contrast, and integrity thresholds.",
                    metric_value=None,
                )
            )
        return issues

    @staticmethod
    def _compute_quality_score(
        dpi_adequate: bool,
        blur_var: float,
        skew_angle_deg: float,
        dynamic_range: float,
        contrast_std: float,
        tamper_score: float,
    ) -> float:
        sharpness_pts = min(30.0, (blur_var / 250.0) * 30.0)
        align_pts = max(0.0, 20.0 - abs(skew_angle_deg) * 2.2)
        contrast_pts = min(25.0, (dynamic_range / 200.0) * 18.0 + (contrast_std / 32.0) * 7.0)
        dpi_pts = 10.0 if dpi_adequate else 4.0
        integrity_pts = max(0.0, 15.0 * (1.0 - tamper_score))
        return float(np.clip(sharpness_pts + align_pts + contrast_pts + dpi_pts + integrity_pts, 0.0, 100.0))
