from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


@dataclass
class DetectorConfig:
    model_path: Path
    input_size: int
    confidence_threshold: float
    nms_iou: float
    class_names: list[str]


class OnnxDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model_version = "unloaded"
        self.ready = False
        self._session = None
        self._input_name = None
        self._input_h = config.input_size
        self._input_w = config.input_size

    def load(self) -> None:
        if ort is None or not self.config.model_path.exists():
            self.model_version = "mock-v0"
            self.ready = False
            return
        self._session = ort.InferenceSession(
            self.config.model_path.as_posix(), providers=["CPUExecutionProvider"]
        )
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        shape = input_meta.shape
        if len(shape) >= 4:
            h_dim = shape[2]
            w_dim = shape[3]
            if isinstance(h_dim, int) and h_dim > 0:
                self._input_h = h_dim
            if isinstance(w_dim, int) and w_dim > 0:
                self._input_w = w_dim
        self.model_version = self.config.model_path.stem
        self.ready = True

    def predict(self, image_bytes: bytes) -> list[dict[str, Any]]:
        if not self.ready:
            return []
        image_tensor, scale = self._preprocess(image_bytes)
        raw_outputs = self._session.run(None, {self._input_name: image_tensor})[0]
        return self._postprocess(raw_outputs, scale)

    def _preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, tuple[float, float]]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ow, oh = img.size
        resized = img.resize((self._input_w, self._input_h))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        sx = ow / self._input_w
        sy = oh / self._input_h
        return arr, (sx, sy)

    def _postprocess(self, outputs: np.ndarray, scale: tuple[float, float]) -> list[dict[str, Any]]:
        if outputs.ndim != 3:
            return []
        preds = outputs[0]
        num_classes = len(self.config.class_names)
        min_no_obj = 4 + num_classes
        min_with_obj = 5 + num_classes

        # Ultralytics ONNX exports can be [1, C, N] or [1, N, C].
        # Normalize to [N, C] for row-wise decoding.
        if preds.shape[0] in (min_no_obj, min_with_obj) and preds.shape[1] > preds.shape[0]:
            preds = preds.T

        detections: list[dict[str, Any]] = []
        for row in preds:
            channels = row.shape[0]
            if channels < min_no_obj:
                continue
            x, y, w, h = row[:4]
            if channels >= min_with_obj:
                obj_conf = float(row[4])
                class_scores = row[5 : 5 + num_classes]
            else:
                obj_conf = 1.0
                class_scores = row[4 : 4 + num_classes]
            if class_scores.size != num_classes:
                continue
            # Some ONNX exports expose raw logits. Convert to probabilities when needed.
            if obj_conf < 0.0 or obj_conf > 1.0:
                obj_conf = float(self._sigmoid(np.array([obj_conf], dtype=np.float32))[0])
            if np.min(class_scores) < 0.0 or np.max(class_scores) > 1.0:
                class_scores = self._sigmoid(class_scores)
            class_id = int(np.argmax(class_scores))
            confidence = obj_conf * float(class_scores[class_id])
            if confidence < self.config.confidence_threshold:
                continue
            sx, sy = scale
            x1 = max(0.0, (x - w / 2) * sx)
            y1 = max(0.0, (y - h / 2) * sy)
            x2 = max(0.0, (x + w / 2) * sx)
            y2 = max(0.0, (y + h / 2) * sy)
            label = self.config.class_names[class_id] if class_id < len(self.config.class_names) else "unknown"
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": label,
                    "confidence": round(confidence, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )
        kept = self._nms(detections, self.config.nms_iou)
        kept.sort(key=lambda item: item["confidence"], reverse=True)
        return kept[:8]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
        ax1, ay1, ax2, ay2 = a["bbox"]["x1"], a["bbox"]["y1"], a["bbox"]["x2"], a["bbox"]["y2"]
        bx1, by1, bx2, by2 = b["bbox"]["x1"], b["bbox"]["y1"], b["bbox"]["x2"], b["bbox"]["y2"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, detections: list[dict[str, Any]], iou_thresh: float) -> list[dict[str, Any]]:
        detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)
        kept: list[dict[str, Any]] = []
        while detections:
            best = detections.pop(0)
            kept.append(best)
            detections = [
                other
                for other in detections
                if not (
                    other["class_id"] == best["class_id"] and self._iou(best, other) > iou_thresh
                )
            ]
        return kept
