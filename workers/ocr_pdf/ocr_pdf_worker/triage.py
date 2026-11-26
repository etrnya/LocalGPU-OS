"""M2-A Local Quality Triage engine for `ocr_pdf` (dual-path metrics, zero cloud calls)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Sequence

TRIAGE_VERSION = "1.0"
IMAGE_COVERAGE_ALGORITHM = "exact_rect_union_v1"
TRIAGE_DISCLAIMER = "Quality triage is a risk indicator and does not guarantee OCR accuracy."


class TriageReason(str, Enum):
    NO_TEXT_DETECTED = "NO_TEXT_DETECTED"
    LOW_TEXT_CONTENT = "LOW_TEXT_CONTENT"
    LOW_AVG_CONFIDENCE = "LOW_AVG_CONFIDENCE"
    HIGH_LOW_CONF_RATIO = "HIGH_LOW_CONF_RATIO"
    INVALID_CONFIDENCE_DATA = "INVALID_CONFIDENCE_DATA"
    IMAGE_HEAVY_PAGE = "IMAGE_HEAVY_PAGE"


@dataclass(frozen=True)
class TriageConfig:
    min_avg_confidence: float = 0.80
    low_conf_block_threshold: float = 0.75
    max_low_conf_ratio: float = 0.20
    low_text_char_threshold: int = 30
    image_heavy_coverage_ratio: float = 0.25
    mixed_text_image_coverage_ratio: float = 0.15
    vector_heavy_drawing_count: int = 120
    vector_heavy_max_chars: int = 150

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_avg_confidence": self.min_avg_confidence,
            "low_conf_block_threshold": self.low_conf_block_threshold,
            "max_low_conf_ratio": self.max_low_conf_ratio,
            "low_text_char_threshold": self.low_text_char_threshold,
            "image_heavy_coverage_ratio": self.image_heavy_coverage_ratio,
            "mixed_text_image_coverage_ratio": self.mixed_text_image_coverage_ratio,
            "vector_heavy_drawing_count": self.vector_heavy_drawing_count,
            "vector_heavy_max_chars": self.vector_heavy_max_chars,
        }


@dataclass(frozen=True)
class PageTriageResult:
    path: str  # "ocr" | "native"
    needs_vision_review: bool
    reasons: tuple[TriageReason, ...]
    metrics: dict[str, Any]
    triage_version: str = TRIAGE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "triage_version": self.triage_version,
            "path": self.path,
            "needs_vision_review": self.needs_vision_review,
            "reasons": [r.value for r in self.reasons],
            "metrics": dict(self.metrics),
        }


def _is_valid_confidence(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    fval = float(value)
    return math.isfinite(fval) and 0.0 <= fval <= 1.0


def compute_exact_rect_union_coverage(
    rects: Iterable[tuple[float, float, float, float]],
    page_width: float,
    page_height: float,
) -> tuple[float, int]:
    """Compute exact union area ratio of rectangles clipped to [0, page_width] x [0, page_height].

    Uses 2D sweep-line coordinate compression so overlapping image bounding boxes
    are never double-counted. Returns (coverage_ratio_in_0_1, valid_rect_count).
    """
    if (
        not math.isfinite(page_width)
        or not math.isfinite(page_height)
        or page_width <= 0.0
        or page_height <= 0.0
    ):
        return 0.0, 0

    clipped: list[tuple[float, float, float, float]] = []
    xs_set: set[float] = set()

    for raw in rects:
        if len(raw) != 4:
            continue
        x0, y0, x1, y1 = raw
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x0, y0, x1, y1)):
            continue
        lx0, lx1 = (x0, x1) if x0 <= x1 else (x1, x0)
        ly0, ly1 = (y0, y1) if y0 <= y1 else (y1, y0)
        cx0 = max(0.0, min(page_width, float(lx0)))
        cx1 = max(0.0, min(page_width, float(lx1)))
        cy0 = max(0.0, min(page_height, float(ly0)))
        cy1 = max(0.0, min(page_height, float(ly1)))
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        clipped.append((cx0, cy0, cx1, cy1))
        xs_set.add(cx0)
        xs_set.add(cx1)

    if not clipped:
        return 0.0, 0

    xs = sorted(xs_set)
    union_area = 0.0

    for idx in range(len(xs) - 1):
        seg_x0, seg_x1 = xs[idx], xs[idx + 1]
        dx = seg_x1 - seg_x0
        if dx <= 0.0:
            continue
        y_intervals: list[tuple[float, float]] = []
        for cx0, cy0, cx1, cy1 in clipped:
            if cx0 <= seg_x0 and cx1 >= seg_x1:
                y_intervals.append((cy0, cy1))
        if not y_intervals:
            continue
        y_intervals.sort()
        merged_y = 0.0
        cur_start, cur_end = y_intervals[0]
        for y0, y1 in y_intervals[1:]:
            if y0 <= cur_end:
                if y1 > cur_end:
                    cur_end = y1
            else:
                merged_y += cur_end - cur_start
                cur_start, cur_end = y0, y1
        merged_y += cur_end - cur_start
        union_area += dx * merged_y

    page_area = page_width * page_height
    ratio = max(0.0, min(1.0, union_area / page_area))
    return round(ratio, 4), len(clipped)


def evaluate_ocr_page_triage(
    blocks: Sequence[tuple[str, Any]],
    char_count: int,
    config: TriageConfig | None = None,
) -> PageTriageResult:
    """Evaluate quality triage for an OCR-processed page.

    Args:
        blocks: Sequence of (text, confidence) pairs.
        char_count: Non-whitespace-trimmed character count of the page.
    """
    cfg = config or TriageConfig()

    valid_confs: list[float] = []
    weighted_sum = 0.0
    weighted_chars = 0
    invalid_conf_count = 0
    valid_block_count = 0

    for text, conf in blocks:
        stripped = str(text).strip() if text is not None else ""
        if not stripped:
            # Blank text blocks are excluded from valid block_count
            continue
        valid_block_count += 1
        if _is_valid_confidence(conf):
            cval = float(conf)
            valid_confs.append(cval)
            clen = len(stripped)
            weighted_sum += cval * clen
            weighted_chars += clen
        else:
            invalid_conf_count += 1

    avg_conf: float | None = (
        round(sum(valid_confs) / len(valid_confs), 4) if valid_confs else None
    )
    char_weighted_avg: float | None = (
        round(weighted_sum / weighted_chars, 4) if weighted_chars > 0 else None
    )
    min_conf: float | None = round(min(valid_confs), 4) if valid_confs else None
    low_conf_blocks = sum(1 for c in valid_confs if c < cfg.low_conf_block_threshold)
    low_conf_ratio = round(low_conf_blocks / len(valid_confs), 4) if valid_confs else 0.0

    reasons: list[TriageReason] = []

    if invalid_conf_count > 0:
        reasons.append(TriageReason.INVALID_CONFIDENCE_DATA)

    # Mutually exclusive text content checks
    if char_count == 0:
        reasons.append(TriageReason.NO_TEXT_DETECTED)
    elif char_count < cfg.low_text_char_threshold:
        reasons.append(TriageReason.LOW_TEXT_CONTENT)

    if (
        (avg_conf is not None and avg_conf < cfg.min_avg_confidence)
        or (char_weighted_avg is not None and char_weighted_avg < cfg.min_avg_confidence)
    ):
        reasons.append(TriageReason.LOW_AVG_CONFIDENCE)

    if valid_confs and low_conf_ratio >= cfg.max_low_conf_ratio:
        reasons.append(TriageReason.HIGH_LOW_CONF_RATIO)

    metrics: dict[str, Any] = {
        "avg_confidence": avg_conf,
        "avg_confidence_method": "block_equal_weight",
        "char_weighted_avg_confidence": char_weighted_avg,
        "min_confidence": min_conf,
        "low_conf_ratio": low_conf_ratio,
        "block_count": valid_block_count,
        "char_count": int(char_count),
        "invalid_confidence_count": invalid_conf_count,
        "image_count": None,
        "image_coverage_ratio": None,
        "image_coverage_algorithm": None,
        "vector_drawing_count": None,
        "has_mixed_text_and_image": None,
    }
    return PageTriageResult(
        path="ocr",
        needs_vision_review=bool(reasons),
        reasons=tuple(reasons),
        metrics=metrics,
    )


def evaluate_native_page_triage(
    char_count: int,
    image_rects: Iterable[tuple[float, float, float, float]],
    page_width: float,
    page_height: float,
    vector_drawing_count: int = 0,
    config: TriageConfig | None = None,
) -> PageTriageResult:
    """Evaluate quality triage for a PyMuPDF native-text PDF page (never fakes OCR confidence)."""
    cfg = config or TriageConfig()

    coverage_ratio, valid_image_count = compute_exact_rect_union_coverage(
        rects=image_rects,
        page_width=page_width,
        page_height=page_height,
    )
    has_mixed = char_count >= 20 and coverage_ratio >= cfg.mixed_text_image_coverage_ratio

    reasons: list[TriageReason] = []

    # Mutually exclusive text content checks
    if char_count == 0:
        reasons.append(TriageReason.NO_TEXT_DETECTED)
    elif char_count < cfg.low_text_char_threshold:
        reasons.append(TriageReason.LOW_TEXT_CONTENT)

    if coverage_ratio >= cfg.image_heavy_coverage_ratio or (
        vector_drawing_count >= cfg.vector_heavy_drawing_count
        and char_count < cfg.vector_heavy_max_chars
    ):
        reasons.append(TriageReason.IMAGE_HEAVY_PAGE)

    metrics: dict[str, Any] = {
        "avg_confidence": None,
        "avg_confidence_method": None,
        "char_weighted_avg_confidence": None,
        "min_confidence": None,
        "low_conf_ratio": None,
        "block_count": None,
        "char_count": int(char_count),
        "invalid_confidence_count": None,
        "image_count": valid_image_count,
        "image_coverage_ratio": coverage_ratio,
        "image_coverage_algorithm": IMAGE_COVERAGE_ALGORITHM,
        "vector_drawing_count": int(vector_drawing_count),
        "has_mixed_text_and_image": has_mixed,
    }
    return PageTriageResult(
        path="native",
        needs_vision_review=bool(reasons),
        reasons=tuple(reasons),
        metrics=metrics,
    )


def build_triage_summary(
    page_triages: Sequence[tuple[int, PageTriageResult]],
    config: TriageConfig | None = None,
    review_images_exported: bool = False,
) -> dict[str, Any]:
    """Build document-level triage_summary with sorted, deduplicated 1-based vision_review_pages."""
    cfg = config or TriageConfig()
    review_pages_set: set[int] = set()
    reason_counts: dict[str, int] = {}

    for page_num, triage in page_triages:
        if triage.needs_vision_review:
            review_pages_set.add(int(page_num))
        for r in triage.reasons:
            reason_counts[r.value] = reason_counts.get(r.value, 0) + 1

    review_pages = sorted(review_pages_set)
    return {
        "triage_version": TRIAGE_VERSION,
        "disclaimer": TRIAGE_DISCLAIMER,
        "total_pages": len(page_triages),
        "review_page_count": len(review_pages),
        "vision_review_pages": review_pages,
        "reason_counts": reason_counts,
        "review_images_exported": review_images_exported,
        "thresholds": cfg.to_dict(),
    }
