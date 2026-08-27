from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "app" / "model" / "yolo11s_ewaste_v2_25class_512.onnx"
DEFAULT_MODEL_CLASSES_PATH = BASE_DIR / "app" / "model" / "ewaste_v2_25class.classes.txt"
DEFAULT_HAZARDS_PATH = BASE_DIR / "app" / "data" / "hazards_rules_v2_25class.json"
DEFAULT_COLLECTION_PATH = BASE_DIR / "app" / "data" / "collection_points_sp.json"
DEFAULT_UPLOADS_DIR = BASE_DIR / "app" / "uploads"
DEFAULT_SQLITE_PATH = BASE_DIR / "app" / "reciclamack.db"


@dataclass(frozen=True)
class Settings:
    app_name: str = "ReciclaMack API"
    model_path: Path = DEFAULT_MODEL_PATH
    model_classes_path: Path = DEFAULT_MODEL_CLASSES_PATH
    hazards_path: Path = DEFAULT_HAZARDS_PATH
    collection_points_path: Path = DEFAULT_COLLECTION_PATH
    uploads_dir: Path = DEFAULT_UPLOADS_DIR
    sqlite_path: Path = DEFAULT_SQLITE_PATH
    image_retention_mode: str = "ttl"
    image_retention_hours: int = 24
    cleanup_interval_seconds: int = 3600
    min_confidence: float = 0.40
    nms_iou: float = 0.45
    input_size: int = 512
    max_upload_mb: int = 10
    cors_allow_origins: tuple[str, ...] = ()
    rate_limit_analyze_per_minute: int = 30
    max_response_detections: int = 8
    dominant_object_gate_enabled: bool = False
    dominant_object_min_area_ratio: float = 0.20
    enable_api_docs: bool = False


def get_settings() -> Settings:
    cors_raw = os.getenv("CORS_ALLOW_ORIGINS", "")
    cors_origins = tuple(
        origin.strip()
        for origin in cors_raw.split(",")
        if origin.strip()
    )
    retention_mode = os.getenv("IMAGE_RETENTION_MODE", "ttl").strip().lower() or "ttl"
    if retention_mode not in {"ttl", "keep"}:
        retention_mode = "ttl"
    return Settings(
        model_path=Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        model_classes_path=Path(os.getenv("MODEL_CLASSES_PATH", str(DEFAULT_MODEL_CLASSES_PATH))),
        hazards_path=Path(os.getenv("HAZARDS_PATH", str(DEFAULT_HAZARDS_PATH))),
        collection_points_path=Path(os.getenv("COLLECTION_POINTS_PATH", str(DEFAULT_COLLECTION_PATH))),
        uploads_dir=Path(os.getenv("UPLOADS_DIR", str(DEFAULT_UPLOADS_DIR))),
        sqlite_path=Path(os.getenv("SQLITE_PATH", str(DEFAULT_SQLITE_PATH))),
        image_retention_mode=retention_mode,
        image_retention_hours=int(os.getenv("IMAGE_RETENTION_HOURS", "24")),
        cleanup_interval_seconds=int(os.getenv("CLEANUP_INTERVAL_SECONDS", "3600")),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.40")),
        nms_iou=float(os.getenv("NMS_IOU", "0.45")),
        input_size=int(os.getenv("INPUT_SIZE", "512")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "10")),
        cors_allow_origins=cors_origins,
        rate_limit_analyze_per_minute=int(os.getenv("RATE_LIMIT_ANALYZE_PER_MINUTE", "30")),
        max_response_detections=int(os.getenv("MAX_RESPONSE_DETECTIONS", "8")),
        dominant_object_gate_enabled=os.getenv("DOMINANT_OBJECT_GATE_ENABLED", "").strip().lower()
        in {"1", "true", "yes", "on"},
        dominant_object_min_area_ratio=float(os.getenv("DOMINANT_OBJECT_MIN_AREA_RATIO", "0.20")),
        enable_api_docs=os.getenv("ENABLE_API_DOCS", "").strip().lower() in {"1", "true", "yes", "on"},
    )
