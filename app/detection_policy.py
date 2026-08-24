from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CROSS_CLASS_DUPLICATE_IOU = 0.90
CROSS_CLASS_MUTUAL_COVERAGE = 0.95


def _box_coordinates(detection: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = detection.get("bbox")
    if not isinstance(bbox, Mapping):
        return None
    try:
        x1, x2 = sorted((float(bbox["x1"]), float(bbox["x2"])))
        y1, y2 = sorted((float(bbox["y1"]), float(bbox["y2"])))
    except (KeyError, TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _overlap_metrics(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> tuple[float, float, float]:
    first_box = _box_coordinates(first)
    second_box = _box_coordinates(second)
    if first_box is None or second_box is None:
        return 0.0, 0.0, 0.0

    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    first_area = (first_x2 - first_x1) * (first_y2 - first_y1)
    second_area = (second_x2 - second_x1) * (second_y2 - second_y1)
    union = first_area + second_area - intersection
    if intersection <= 0.0 or union <= 0.0:
        return 0.0, 0.0, 0.0
    return intersection / union, intersection / first_area, intersection / second_area


def collapse_cross_class_duplicates(
    detections: Sequence[dict[str, object]],
    *,
    iou_threshold: float = CROSS_CLASS_DUPLICATE_IOU,
    mutual_coverage_threshold: float = CROSS_CLASS_MUTUAL_COVERAGE,
) -> list[dict[str, object]]:
    """Keep the strongest result for near-identical boxes with different classes."""
    ranked = sorted(
        detections,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )
    retained: list[dict[str, object]] = []
    for candidate in ranked:
        candidate_class = str(candidate.get("class_name", "")).lower()
        is_duplicate = False
        for stronger in retained:
            if candidate_class == str(stronger.get("class_name", "")).lower():
                continue
            iou, candidate_coverage, stronger_coverage = _overlap_metrics(candidate, stronger)
            if (
                iou >= iou_threshold
                and candidate_coverage >= mutual_coverage_threshold
                and stronger_coverage >= mutual_coverage_threshold
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            retained.append(candidate)
    return retained
