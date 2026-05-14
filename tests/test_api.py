from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import _load_model_class_names, create_app


def _settings(tmp_path: Path) -> Settings:
    hazards_path = Path(__file__).resolve().parents[1] / "app" / "data" / "hazards_rules.json"
    collection_path = Path(__file__).resolve().parents[1] / "app" / "data" / "collection_points_sp.json"
    return Settings(
        model_path=tmp_path / "missing.onnx",
        hazards_path=hazards_path,
        collection_points_path=collection_path,
        uploads_dir=tmp_path / "uploads",
        sqlite_path=tmp_path / "requests.db",
        cleanup_interval_seconds=9999,
    )


def test_health_and_classes(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    health = client.get("/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert "content_version" in body

    classes = client.get("/v1/classes")
    assert classes.status_code == 200
    payload = classes.json()
    assert payload["classes"]
    assert all("class_name" in item for item in payload["classes"])
    assert all("display_label_pt_br" in item for item in payload["classes"])


def test_v1_model_class_file_maps_to_content_class_names(tmp_path: Path) -> None:
    classes_path = tmp_path / "v1.classes.txt"
    classes_path.write_text("bateria\ncelular\nplaca_eletronica\n", encoding="utf-8")

    assert _load_model_class_names(classes_path) == [
        "battery",
        "mobile_phone_tablet",
        "computer_part",
    ]


def test_analyze_image_rejects_non_image(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_analyze_image_rejects_oversized_upload(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    client = TestClient(app)

    oversized = b"x" * (settings.max_upload_mb * 1024 * 1024 + 1)
    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("large.jpg", oversized, "image/jpeg")},
    )
    assert resp.status_code == 413


def test_analyze_image_rate_limit_is_opt_in(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = Settings(
        model_path=settings.model_path,
        model_classes_path=settings.model_classes_path,
        hazards_path=settings.hazards_path,
        collection_points_path=settings.collection_points_path,
        uploads_dir=settings.uploads_dir,
        sqlite_path=settings.sqlite_path,
        image_retention_hours=settings.image_retention_hours,
        cleanup_interval_seconds=settings.cleanup_interval_seconds,
        min_confidence=settings.min_confidence,
        nms_iou=settings.nms_iou,
        input_size=settings.input_size,
        max_upload_mb=settings.max_upload_mb,
        cors_allow_origins=settings.cors_allow_origins,
        rate_limit_analyze_per_minute=1,
        max_response_classes=settings.max_response_classes,
    )
    app = create_app(settings)
    client = TestClient(app)

    first = client.post(
        "/v1/analyze-image",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    second = client.post(
        "/v1/analyze-image",
        files={"file": ("notes.txt", b"hello again", "text/plain")},
    )

    assert first.status_code == 400
    assert second.status_code == 429


def test_analyze_image_response_shape(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    # Mock detector because model is intentionally absent in unit tests.
    app.dependency_overrides = {}
    image = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00"
        b"\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    resp = client.post("/v1/analyze-image", files={"file": ("img.png", image, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "request_id" in body
    assert "model_version" in body
    assert isinstance(body["detections"], list)
    assert isinstance(body["guidance"], list)
    assert "uncertainty_flag" in body
    for item in body["detections"]:
        assert "display_label_pt_br" in item
    for item in body["guidance"]:
        assert "display_label_pt_br" in item


def test_analyze_image_suppresses_overlapping_cross_class_detections(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    app.state.detector.ready = True
    app.state.detector.model_version = "test-model"
    app.state.detector.predict = lambda _payload: [
        {
            "class_id": 0,
            "class_name": "battery",
            "confidence": 0.96,
            "bbox": {"x1": 10, "y1": 10, "x2": 110, "y2": 110},
        },
        {
            "class_id": 25,
            "class_name": "smart_watch",
            "confidence": 0.95,
            "bbox": {"x1": 12, "y1": 12, "x2": 108, "y2": 108},
        },
        {
            "class_id": 999,
            "class_name": "unsupported_future_class",
            "confidence": 0.99,
            "bbox": {"x1": 200, "y1": 200, "x2": 260, "y2": 260},
        },
    ]

    image = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00"
        b"\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    resp = client.post("/v1/analyze-image", files={"file": ("img.png", image, "image/png")})

    assert resp.status_code == 200
    body = resp.json()
    assert [item["class_name"] for item in body["detections"]] == ["battery"]
    assert [item["class_name"] for item in body["guidance"]] == ["battery"]


def test_analyze_image_v1_caps_response_to_top_class(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    app.state.detector.ready = True
    app.state.detector.model_version = "test-model"
    app.state.detector.predict = lambda _payload: [
        {
            "class_id": 0,
            "class_name": "battery",
            "confidence": 0.96,
            "bbox": {"x1": 10, "y1": 10, "x2": 110, "y2": 110},
        },
        {
            "class_id": 25,
            "class_name": "smart_watch",
            "confidence": 0.95,
            "bbox": {"x1": 200, "y1": 200, "x2": 260, "y2": 260},
        },
    ]

    image = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00"
        b"\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    resp = client.post("/v1/analyze-image", files={"file": ("img.png", image, "image/png")})

    assert resp.status_code == 200
    body = resp.json()
    assert [item["class_name"] for item in body["detections"]] == ["battery"]
    assert [item["class_name"] for item in body["guidance"]] == ["battery"]
