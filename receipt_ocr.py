
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
import unicodedata
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

_PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC = "٠١٢٣٤٥٦٧٨٩"
_LATIN = "0123456789"
_TO_LATIN = str.maketrans(_PERSIAN + _ARABIC, _LATIN * 2)
_TO_PERSIAN = str.maketrans(_LATIN, _PERSIAN)
_LETTER_TABLE = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})
_MIN_CONFIDENCE = 0.5
_NUMBER_REGEX = re.compile(r"[0-9]+")


def normalize_digits(text: str, style: str = "persian") -> str:
    """Normalize Persian/Arabic/ASCII digits without guessing their values.

    style='persian': user-visible output: ۱۲۳۴۵
    style='latin': machine-readable output: 12345
    """
    if style not in ("persian", "latin"):
        raise ValueError("style must be 'persian' or 'latin'")
    text = str(text).translate(_TO_LATIN)
    return text.translate(_TO_PERSIAN) if style == "persian" else text


def normalize_receipt_text(text: str, style: str = "persian") -> str:
    """Conservative cleanup. Unlike upstream normalizer, never drop short lines."""
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.translate(_LETTER_TABLE)
    return normalize_digits(text, style)


@dataclass
class ReceiptLine:
    text: str
    score: float
    bbox: list[list[float]]
    numeric_candidates: list[dict[str, Any]] = field(default_factory=list)
    needs_review: bool = False
    review_reason: str = ""


@dataclass
class ReceiptResult:
    text: str
    latin_text: str
    lines: list[ReceiptLine]
    needs_review: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_raw_lines(result: Any, digit_style: str) -> list[ReceiptLine]:
    if result is None:
        return []
    texts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    scores = getattr(result, "scores", None)
    if texts is None:
        return []
    lines: list[ReceiptLine] = []
    for i, raw in enumerate(texts):
        raw = str(raw or "").strip()
        score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        if score < _MIN_CONFIDENCE or not raw or not any(c.isalnum() for c in raw):
            continue
        points = (
            [[float(x), float(y)] for x, y in boxes[i]]
            if boxes is not None and i < len(boxes)
            else [[0.0, float(i)], [1.0, float(i)], [1.0, float(i) + 1], [0.0, float(i) + 1]]
        )
        lines.append(ReceiptLine(
            text=normalize_receipt_text(raw, digit_style),
            score=round(score, 4),
            bbox=points,
        ))
    # Reading order for RTL page; numeric strings retain their internal order.
    lines.sort(key=lambda line: (min(p[1] for p in line.bbox),
                                 -max(p[0] for p in line.bbox)))
    return lines


def _numeric_groups(text: str) -> list[str]:
    return _NUMBER_REGEX.findall(normalize_digits(text, "latin"))


def _should_check(line: ReceiptLine) -> bool:
    groups = _numeric_groups(line.text)
    if not groups or max(map(len, groups)) < 3:
        return False
    # Numeric-heavy fields and lower-confidence mixed fields benefit most.
    alnum_count = sum(c.isalnum() for c in line.text)
    digit_count = sum(c.isdigit() for c in line.text)
    numeric_heavy = alnum_count > 0 and (digit_count / alnum_count) >= 0.50
    return numeric_heavy or line.score < 0.88


def _rectify_box(image_bgr: np.ndarray, points: list[list[float]]) -> Image.Image | None:
    """Perspective-correct a detected 4-corner text line before retrying."""
    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2):
        return None
    tl, tr, br, bl = quad
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 9 or height < 5 or width > 6000 or height > 2000:
        return None
    dest = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(quad, dest)
    warped = cv2.warpPerspective(image_bgr, matrix, (width, height),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # White-ish border prevents characters at the edge from being cut off.
    vpad = max(4, int(height * 0.22))
    hpad = max(6, int(height * 0.35))
    warped = cv2.copyMakeBorder(warped, vpad, vpad, hpad, hpad,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def _retry_variants(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    scale = min(2.5, max(1.5, 68 / max(height, 1)))
    new_size = (int(width * scale), int(height * scale))
    upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
    contrasted = ImageEnhance.Contrast(ImageOps.autocontrast(upscaled.convert("L"))).enhance(1.3)
    sharpened = contrasted.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
    return [upscaled, sharpened.convert("RGB")]


def _check_line_numbers(
    line: ReceiptLine,
    image_bgr: np.ndarray,
    engine: Any,
    infer_lock: Any,
    digit_style: str,
    cancel_check: Callable[[], bool] | None,
) -> None:
    crop = _rectify_box(image_bgr, line.bbox)
    if crop is None:
        return
    baseline = _numeric_groups(line.text)
    possible: dict[tuple[str, ...], dict[str, Any]] = {}
    for variant in _retry_variants(crop):
        if cancel_check is not None and cancel_check():
            from ocr_utils import OCRCancelled
            raise OCRCancelled()
        # RapidOCR interprets ndarray input as BGR (OpenCV convention).
        arr_bgr = cv2.cvtColor(np.asarray(variant), cv2.COLOR_RGB2BGR)
        with infer_lock:
            # Critical: RapidOCR updates use_det/use_cls persistently per call.
            # Restore them so subsequent full-page OCR still detects text.
            saved = {name: getattr(engine, name) for name in
                     ("use_det", "use_cls", "use_rec") if hasattr(engine, name)}
            try:
                retry_result = engine(arr_bgr, use_det=False, use_cls=False, use_rec=True)
            finally:
                if saved:
                    if hasattr(engine, "update_params"):
                        engine.update_params(**saved)
                    else:
                        for name, value in saved.items():
                            setattr(engine, name, value)
        texts = getattr(retry_result, "txts", None) if retry_result is not None else None
        scores = getattr(retry_result, "scores", None) if retry_result is not None else None
        if texts is None or len(texts) == 0:
            continue
        value = normalize_receipt_text(" ".join(map(str, texts)), digit_style)
        groups = _numeric_groups(value)
        if not groups:
            continue
        # Compare only if number group count and width are compatible.
        if len(groups) != len(baseline) or [len(s) for s in groups] != [len(s) for s in baseline]:
            continue
        conf = min((float(s) for s in scores), default=0.0) if scores is not None else 0.0
        key = tuple(groups)
        if key in possible:
            possible[key]["votes"] += 1
            possible[key]["score"] = max(possible[key]["score"], round(conf, 4))
        else:
            possible[key] = {"value": value, "digits": groups,
                             "score": round(conf, 4), "votes": 1}
    line.numeric_candidates = sorted(possible.values(), key=lambda c: (-c["votes"], -c["score"]))
    differs = any(c["digits"] != baseline and c["score"] >= 0.55
                  for c in line.numeric_candidates)
    if differs:
        line.needs_review = True
        line.review_reason = "OCR retries disagree on one or more digit values"
    elif line.score < 0.70:
        line.needs_review = True
        line.review_reason = "Low recognition confidence for a numeric field"


def analyze_receipt(
    image: Image.Image,
    lang: str = "fas+eng",
    mode: str = "accurate",
    digit_style: str = "persian",
    check_numbers: bool = True,
    max_checks: int = 10,
    progress_cb: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> ReceiptResult:
    """OCR a receipt with numeric preservation and optional uncertainty checks.

    IMPORTANT: The second pass is a diagnostic; it does not silently overwrite
    amounts, dates, or transaction IDs. Human/bank validation is still required.
    """
    if digit_style not in ("persian", "latin"):
        raise ValueError("digit_style must be 'persian' or 'latin'")
    if max_checks < 0:
        raise ValueError("max_checks must be >= 0")
    from ocr_utils import _infer_lock, _prepare_base, _run_engine, get_engine

    def report(value: float, stage: str) -> None:
        if progress_cb:
            progress_cb(value, stage)

    report(0.05, "prepare")
    engine = get_engine("en" if lang.strip().lower() in ("eng", "en", "english") else "fa")
    base = _prepare_base(image, mode)
    base_bgr = cv2.cvtColor(np.asarray(base.convert("RGB")), cv2.COLOR_RGB2BGR)
    report(0.20, "read text")
    result = _run_engine(engine, base_bgr, cancel_check, return_word_box=False)
    lines = _get_raw_lines(result, digit_style)
    report(0.65, "text recognized")
    if check_numbers:
        targets = [line for line in lines if _should_check(line)][:max_checks]
        for index, line in enumerate(targets):
            _check_line_numbers(line, base_bgr, engine, _infer_lock, digit_style, cancel_check)
            report(0.65 + 0.33 * ((index + 1) / max(1, len(targets))), "checking numbers")
    text = "\n".join(line.text for line in lines)
    report(1.0, "done")
    return ReceiptResult(
        text=text,
        latin_text=normalize_digits(text, "latin"),
        lines=lines,
        needs_review=any(line.needs_review for line in lines),
    )


def run_ocr(
    image: Image.Image,
    lang: str = "fas+eng",
    mode: str = "accurate",
    progress_cb: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    **kwargs: Any,
) -> str:
    """Drop-in alternative for project's run_ocr(); still returns plain text."""
    return analyze_receipt(
        image=image, lang=lang, mode=mode,
        progress_cb=progress_cb, cancel_check=cancel_check,
        digit_style=kwargs.pop("digit_style", "persian"),
        check_numbers=kwargs.pop("check_numbers", True),
        max_checks=kwargs.pop("max_checks", 10),
    ).text
