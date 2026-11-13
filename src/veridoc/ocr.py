"""Tesseract OCR extractor and pluggable OCR backend protocol."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol, Union, runtime_checkable

import pytesseract
from PIL import Image
from pytesseract import Output

from veridoc.models import OCRResult, WordBox

OCRImageInput = Union[bytes, str, Path, Image.Image]


@runtime_checkable
class OCRBackend(Protocol):
    """Protocol for pluggable OCR engines (Tesseract or test fakes)."""

    def extract(self, image_input: OCRImageInput) -> OCRResult:
        """Extract text, word bounding boxes, and confidence metrics from an image."""
        ...


def check_tesseract_available(tesseract_cmd: str | None = None) -> tuple[bool, str | None]:
    """Return ``(is_available, version_string)`` for the system Tesseract binary."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        version = pytesseract.get_tesseract_version()
        return True, str(version)
    except Exception:
        return False, None


class TesseractExtractor:
    """Real Tesseract OCR wrapper returning word boxes, confidences, and cleaned text."""

    def __init__(
        self,
        lang: str = "eng",
        psm: int = 3,
        oem: int = 3,
        low_conf_threshold: float = 70.0,
        tesseract_cmd: str | None = None,
    ) -> None:
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.low_conf_threshold = low_conf_threshold
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract(self, image_input: OCRImageInput) -> OCRResult:
        """Extract structured OCR words, confidences, and line-ordered text."""
        pil_img = self._to_pil(image_input)
        config = f"--oem {self.oem} --psm {self.psm}"
        data = pytesseract.image_to_data(
            pil_img,
            lang=self.lang,
            config=config,
            output_type=Output.DICT,
        )

        words: list[WordBox] = []
        lines: dict[tuple[int, int, int], list[str]] = {}

        n_items = len(data.get("text", []))
        for idx in range(n_items):
            raw_token = str(data["text"][idx] or "").strip()
            if not raw_token:
                continue

            raw_conf = data["conf"][idx]
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = -1.0

            if conf < 0.0:
                continue

            block_num = int(data.get("block_num", [0] * n_items)[idx])
            par_num = int(data.get("par_num", [0] * n_items)[idx])
            line_num = int(data.get("line_num", [0] * n_items)[idx])
            left = int(data.get("left", [0] * n_items)[idx])
            top = int(data.get("top", [0] * n_items)[idx])
            width = int(data.get("width", [0] * n_items)[idx])
            height = int(data.get("height", [0] * n_items)[idx])

            clamped_conf = max(0.0, min(100.0, conf))
            words.append(
                WordBox(
                    text=raw_token,
                    confidence=round(clamped_conf, 2),
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    line_num=line_num,
                    block_num=block_num,
                )
            )
            lines.setdefault((block_num, par_num, line_num), []).append(raw_token)

        if not words:
            return OCRResult(
                text="",
                words=[],
                word_count=0,
                mean_confidence=0.0,
                low_confidence_words=[],
                engine="tesseract",
            )

        ordered_lines = [" ".join(lines[key]) for key in sorted(lines.keys())]
        extracted_text = "\n".join(ordered_lines).strip()
        mean_conf = sum(w.confidence for w in words) / len(words)
        low_conf_words = [w.text for w in words if w.confidence < self.low_conf_threshold]

        return OCRResult(
            text=extracted_text,
            words=words,
            word_count=len(words),
            mean_confidence=round(mean_conf, 2),
            low_confidence_words=low_conf_words,
            engine="tesseract",
        )

    @staticmethod
    def _to_pil(image_input: OCRImageInput) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, (str, Path)):
            with Image.open(Path(image_input)) as img:
                return img.convert("RGB")
        if isinstance(image_input, bytes):
            if not image_input:
                raise ValueError("Empty image bytes passed to OCR extractor.")
            with Image.open(io.BytesIO(image_input)) as img:
                return img.convert("RGB")
        raise TypeError(f"Unsupported OCR input type: {type(image_input).__name__}")
