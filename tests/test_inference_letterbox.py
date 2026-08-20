from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.inference import DetectorConfig, LetterboxTransform, OnnxDetector


def detector(input_size: int = 8) -> OnnxDetector:
    return OnnxDetector(
        DetectorConfig(Path("unused.onnx"), input_size, 0.40, 0.45, ["item"])
    )


def image_bytes(
    size: tuple[int, int],
    color: tuple[int, int, int] = (10, 20, 30),
    orientation: int | None = None,
) -> bytes:
    image = Image.new("RGB", size, color)
    output = io.BytesIO()
    if orientation is None:
        image.save(output, format="PNG")
    else:
        exif = image.getexif()
        exif[274] = orientation
        image.save(output, format="JPEG", exif=exif)
    return output.getvalue()


@pytest.mark.parametrize(
    ("size", "expected_scale", "expected_pad"),
    [
        ((4, 2), 2.0, (0, 2)),
        ((2, 4), 2.0, (2, 0)),
        ((4, 4), 2.0, (0, 0)),
        ((5, 3), 1.6, (0, 1)),
    ],
)
def test_preprocess_uses_centered_letterbox(
    size: tuple[int, int],
    expected_scale: float,
    expected_pad: tuple[int, int],
) -> None:
    tensor, transform = detector()._preprocess(image_bytes(size))

    assert tensor.shape == (1, 3, 8, 8)
    assert transform.scale == pytest.approx(expected_scale)
    assert (transform.pad_x, transform.pad_y) == expected_pad
    assert (transform.original_width, transform.original_height) == size

    pixels = np.transpose(tensor[0], (1, 2, 0)) * 255.0
    if transform.pad_y:
        assert pixels[0, 0] == pytest.approx([114, 114, 114])
    if transform.pad_x:
        assert pixels[0, 0] == pytest.approx([114, 114, 114])


def test_preprocess_applies_exif_orientation_before_letterbox() -> None:
    _, transform = detector()._preprocess(image_bytes((4, 2), orientation=6))

    assert (transform.original_width, transform.original_height) == (2, 4)
    assert (transform.pad_x, transform.pad_y) == (2, 0)


def output_row(
    x: float,
    y: float,
    width: float,
    height: float,
    confidence: float = 0.90,
) -> np.ndarray:
    return np.array(
        [[[x, y, width, height, confidence]]],
        dtype=np.float32,
    )


def test_postprocess_removes_padding_and_clamps_to_source_image() -> None:
    configured = detector(input_size=512)
    transform = LetterboxTransform(5.12, 0, 128, 100, 50)

    predictions = configured._postprocess(
        output_row(256.0, 256.0, 409.6, 204.8), transform
    )

    assert len(predictions) == 1
    assert predictions[0]["bbox"] == pytest.approx(
        {"x1": 10.0, "y1": 5.0, "x2": 90.0, "y2": 45.0}, abs=1e-4
    )


def test_postprocess_removes_boxes_that_exist_only_in_padding() -> None:
    configured = detector(input_size=512)
    transform = LetterboxTransform(5.12, 0, 128, 100, 50)

    predictions = configured._postprocess(
        output_row(256.0, 50.0, 100.0, 40.0), transform
    )

    assert predictions == []
