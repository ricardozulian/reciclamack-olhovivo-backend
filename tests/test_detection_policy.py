from __future__ import annotations

from app.detection_policy import collapse_cross_class_duplicates, retain_dominant_detections
from app.main import _filter_v1_detections


def _detection(
    class_name: str,
    confidence: float,
    box: tuple[float, float, float, float],
) -> dict[str, object]:
    x1, y1, x2, y2 = box
    return {
        "class_id": 0,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def test_collapses_near_identical_cross_class_boxes_to_highest_confidence() -> None:
    detections = [
        _detection("av_equipment", 0.43, (380.7, 465.3, 3627.5, 2348.1)),
        _detection("portable_music_player", 0.47, (381.7, 460.7, 3629.2, 2346.6)),
    ]

    result = collapse_cross_class_duplicates(detections)

    assert [item["class_name"] for item in result] == ["portable_music_player"]


def test_does_not_collapse_same_class_boxes() -> None:
    detections = [
        _detection("battery", 0.90, (0, 0, 100, 100)),
        _detection("battery", 0.80, (1, 1, 99, 99)),
    ]

    assert collapse_cross_class_duplicates(detections) == detections


def test_does_not_collapse_distinct_overlapping_cross_class_objects() -> None:
    detections = [
        _detection("laptop", 0.90, (0, 0, 100, 100)),
        _detection("keyboard", 0.80, (20, 20, 100, 100)),
    ]

    assert collapse_cross_class_duplicates(detections) == detections


def test_does_not_collapse_a_smaller_object_nested_in_a_larger_object() -> None:
    detections = [
        _detection("flat_monitor", 0.90, (0, 0, 100, 100)),
        _detection("mouse", 0.80, (25, 25, 75, 75)),
    ]

    assert collapse_cross_class_duplicates(detections) == detections


def test_response_limit_is_applied_after_cross_class_collapse() -> None:
    detections = [
        _detection("portable_music_player", 0.99, (0, 0, 100, 100)),
        _detection("av_equipment", 0.98, (1, 1, 99, 99)),
        _detection("battery", 0.97, (110, 0, 150, 40)),
        _detection("cable", 0.96, (160, 0, 200, 40)),
    ]

    result = _filter_v1_detections(
        detections,
        {"portable_music_player", "av_equipment", "battery", "cable"},
        min_confidence=0.40,
        max_response_detections=3,
    )

    assert [item["class_name"] for item in result] == [
        "portable_music_player",
        "battery",
        "cable",
    ]


def test_dominant_gate_removes_small_totem_false_positives() -> None:
    dominant = _detection("mobile_phone_tablet", 0.90, (10, 10, 90, 90))
    large_noise = _detection("cable", 0.80, (80, 0, 100, 75))
    small_noise = _detection("battery", 0.70, (0, 0, 10, 10))

    result = retain_dominant_detections(
        [dominant, large_noise, small_noise],
        (100, 100),
        min_dominant_area_ratio=0.20,
        min_relative_area_ratio=0.25,
        min_absolute_area_ratio=0.05,
    )

    assert result == [dominant]


def test_dominant_gate_keeps_multiple_large_objects() -> None:
    dominant = _detection("flat_monitor", 0.90, (0, 0, 100, 50))
    secondary = _detection("keyboard", 0.80, (0, 60, 100, 75))

    result = retain_dominant_detections(
        [dominant, secondary],
        (100, 100),
        min_dominant_area_ratio=0.20,
        min_relative_area_ratio=0.25,
        min_absolute_area_ratio=0.05,
    )

    assert result == [dominant, secondary]


def test_dominant_gate_requests_a_new_image_without_a_large_object() -> None:
    detections = [_detection("battery", 0.90, (0, 0, 40, 40))]

    result = retain_dominant_detections(
        detections,
        (100, 100),
        min_dominant_area_ratio=0.20,
        min_relative_area_ratio=0.25,
        min_absolute_area_ratio=0.05,
    )

    assert result == []
