from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.inference import DetectorConfig, LetterboxTransform, OnnxDetector


def detector(input_size: int = 8) -> OnnxDetector:
    return OnnxDetector(
        DetectorConfig(Path("unused.onnx"), input_size, 0.40, 0.45, ["item"])
    )


class FakeInput:
    name = "images"
    shape = [1, 3, 512, 512]


class FakeSession:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]


def test_load_uses_explicit_model_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test")
    monkeypatch.setattr(
        "app.inference.ort",
        SimpleNamespace(InferenceSession=FakeSession),
    )
    configured = OnnxDetector(
        DetectorConfig(
            model_path,
            512,
            0.40,
            0.45,
            ["item"],
            model_version="v2_1_2_letterbox_enhanced_adamw_e0_512_yolo11s_epoch50",
        )
    )

    configured.load()

    assert configured.model_version == (
        "v2_1_2_letterbox_enhanced_adamw_e0_512_yolo11s_epoch50"
    )


def test_load_uses_filename_when_model_version_is_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test")
    monkeypatch.setattr(
        "app.inference.ort",
        SimpleNamespace(InferenceSession=FakeSession),
    )
    configured = OnnxDetector(
        DetectorConfig(model_path, 512, 0.40, 0.45, ["item"])
    )

    configured.load()

    assert configured.model_version == "model"


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
