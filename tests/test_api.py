from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

from fastapi import Request
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import _client_ip, _load_model_class_names, create_app

V2_25CLASS_MODEL_CLASSES = [
    "battery",
    "cable",
    "mobile_phone_tablet",
    "printer_multifunction",
    "flat_monitor",
    "mouse",
    "laptop",
    "computer_part",
    "portable_music_player",
    "network_device",
    "landline_telephone",
    "crt_monitor",
    "usb_stick",
    "ink_toner_cartridge",
    "camera",
    "keyboard",
    "power_source_charger",
    "remote",
    "power_tool",
    "clock_radio",
    "headset",
    "microphone",
    "smart_watch",
    "home_appliance",
    "av_equipment",
]


def _settings(tmp_path: Path) -> Settings:
    app_path = Path(__file__).resolve().parents[1] / "app"
    hazards_path = app_path / "data" / "hazards_rules.json"
    collection_path = app_path / "data" / "collection_points_sp.json"
    return Settings(
        model_path=tmp_path / "missing.onnx",
        model_classes_path=Path(__file__).resolve().parent / "fixtures_yolo11n_ewaste_v1.classes.txt",
        hazards_path=hazards_path,
        collection_points_path=collection_path,
        uploads_dir=tmp_path / "uploads",
        sqlite_path=tmp_path / "requests.db",
        image_retention_mode="ttl",
        cleanup_interval_seconds=9999,
        input_size=640,
    )


def _png(width: int = 1, height: int = 1) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(output, format="PNG")
    return output.getvalue()


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


def test_api_docs_are_disabled_by_default(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_api_responses_include_security_headers(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    resp = client.get("/v1/health")

    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert resp.headers["permissions-policy"] == "camera=(self), microphone=()"


def test_classes_endpoint_uses_active_v1_model_classes(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    resp = client.get("/v1/classes")

    assert resp.status_code == 200
    class_names = [item["class_name"] for item in resp.json()["classes"]]
    assert class_names == [
        "battery",
        "cable",
        "capacitor",
        "mobile_phone_tablet",
        "printer_multifunction",
        "flat_monitor",
        "mouse",
        "laptop",
        "computer_part",
        "portable_music_player",
        "network_device",
        "landline_telephone",
        "crt_monitor",
        "usb_stick",
    ]
    assert "smart_watch" not in class_names
    assert "camera" not in class_names


def test_classes_endpoint_uses_v2_25class_runtime_pairing(tmp_path: Path) -> None:
    base = Path(__file__).resolve().parents[1] / "app"
    settings = replace(
        _settings(tmp_path),
        model_classes_path=(
            base
            / "model"
            / "v2_1_2_letterbox_enhanced_adamw_e0_512_yolo11s_epoch50.classes.txt"
        ),
        hazards_path=base / "data" / "hazards_rules_v2_25class.json",
        input_size=512,
    )
    app = create_app(settings)
    client = TestClient(app)

    resp = client.get("/v1/classes")

    assert resp.status_code == 200
    classes = resp.json()["classes"]
    class_names = [item["class_name"] for item in classes]
    assert class_names == V2_25CLASS_MODEL_CLASSES
    assert "capacitor" not in class_names
    av_equipment = next(item for item in classes if item["class_name"] == "av_equipment")
    assert av_equipment["display_label_pt_br"] == "equipamento de áudio e vídeo"
    assert app.state.detector.config.input_size == 512


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


def test_analyze_image_rejects_spoofed_image_content_type(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("fake.jpg", b"not an image", "image/jpeg")},
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
        image_retention_mode=settings.image_retention_mode,
        image_retention_hours=settings.image_retention_hours,
        cleanup_interval_seconds=settings.cleanup_interval_seconds,
        min_confidence=settings.min_confidence,
        nms_iou=settings.nms_iou,
        input_size=settings.input_size,
        max_upload_mb=settings.max_upload_mb,
        cors_allow_origins=settings.cors_allow_origins,
        rate_limit_analyze_per_minute=1,
        max_response_detections=settings.max_response_detections,
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


def _request_with_forwarding_headers(**headers: str) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [
                (name.replace("_", "-").encode("ascii"), value.encode("ascii"))
                for name, value in headers.items()
            ],
            "client": ("172.18.0.2", 12345),
        }
    )


def test_rate_limit_client_ip_uses_proxy_appended_address() -> None:
    request = _request_with_forwarding_headers(
        x_forwarded_for="10.0.0.9, 172.31.0.3, 8.8.8.8"
    )

    assert _client_ip(request) == "8.8.8.8"


def test_rate_limit_client_ip_keeps_lan_client_address() -> None:
    request = _request_with_forwarding_headers(
        x_forwarded_for="8.8.8.8, 192.168.1.44"
    )

    assert _client_ip(request) == "192.168.1.44"


def test_rate_limit_client_ip_prefers_cloudflare_connecting_ip() -> None:
    request = _request_with_forwarding_headers(
        cf_connecting_ip="2001:4860:4860::8888",
        x_forwarded_for="1.2.3.4, 172.18.0.3",
    )

    assert _client_ip(request) == "2001:4860:4860::8888"


def test_rate_limit_client_ip_rejects_invalid_cloudflare_address() -> None:
    request = _request_with_forwarding_headers(
        cf_connecting_ip="not-an-ip",
        x_forwarded_for="203.0.113.8, 192.168.1.44",
    )

    assert _client_ip(request) == "192.168.1.44"


def test_analyze_image_response_shape(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    # Mock detector because model is intentionally absent in unit tests.
    app.dependency_overrides = {}
    image = _png()
    resp = client.post("/v1/analyze-image", files={"file": ("img.png", image, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "request_id" in body
    assert "model_version" in body
    assert body["image_width"] == 1
    assert body["image_height"] == 1
    assert isinstance(body["detections"], list)
    assert isinstance(body["guidance"], list)
    assert "uncertainty_flag" in body
    for item in body["detections"]:
        assert "display_label_pt_br" in item
    for item in body["guidance"]:
        assert "display_label_pt_br" in item


def test_analyze_image_probe_does_not_persist_files_or_metadata(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    client = TestClient(app)

    app.state.detector.ready = True
    app.state.detector.model_version = "test-model"
    app.state.detector.predict = lambda _payload: [
        {
            "class_id": 0,
            "class_name": "battery",
            "confidence": 0.96,
            "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        }
    ]

    resp = client.post(
        "/v1/analyze-image?persist=false",
        files={"file": ("probe.png", _png(), "image/png")},
    )

    assert resp.status_code == 200
    assert resp.json()["detections"][0]["class_name"] == "battery"
    assert app.state.repository.fetch_request(resp.json()["request_id"]) is None
    assert list(settings.uploads_dir.iterdir()) == []


def test_analyze_image_writes_dataset_label_for_classified_upload(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    app.state.detector.ready = True
    app.state.detector.model_version = "test-model"
    app.state.detector.predict = lambda _payload: [
        {
            "class_id": 0,
            "class_name": "battery",
            "confidence": 0.96,
            "bbox": {"x1": 10, "y1": 20, "x2": 50, "y2": 80},
        }
    ]

    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("img.png", _png(width=100, height=200), "image/png")},
    )

    assert resp.status_code == 200
    row = app.state.repository.fetch_request(resp.json()["request_id"])
    assert row is not None
    label_path = Path(row["stored_path"]).with_suffix(".txt")
    assert label_path.read_text(encoding="utf-8") == (
        "0 0.300000 0.250000 0.400000 0.300000\n"
    )


def test_analyze_image_writes_null_label_when_unclassified(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    app.state.detector.ready = True
    app.state.detector.model_version = "test-model"
    app.state.detector.predict = lambda _payload: []

    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("img.png", _png(width=100, height=200), "image/png")},
    )

    assert resp.status_code == 200
    row = app.state.repository.fetch_request(resp.json()["request_id"])
    assert row is not None
    assert Path(row["stored_path"]).with_suffix(".txt").read_text(encoding="utf-8") == "null\n"


def test_analyze_image_writes_null_label_when_inference_fails(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    def fail_predict(_payload: bytes) -> list[dict[str, object]]:
        raise RuntimeError("boom")

    app.state.detector.ready = True
    app.state.detector.model_version = "test-model"
    app.state.detector.predict = fail_predict

    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("img.png", _png(width=100, height=200), "image/png")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["uncertainty_flag"] is True
    assert body["detections"] == []
    row = app.state.repository.fetch_request(body["request_id"])
    assert row is not None
    assert Path(row["stored_path"]).with_suffix(".txt").read_text(encoding="utf-8") == "null\n"


def test_analyze_image_persists_keep_retention_mode(tmp_path: Path) -> None:
    settings = Settings(
        model_path=tmp_path / "missing.onnx",
        model_classes_path=_settings(tmp_path).model_classes_path,
        hazards_path=_settings(tmp_path).hazards_path,
        collection_points_path=_settings(tmp_path).collection_points_path,
        uploads_dir=tmp_path / "uploads",
        sqlite_path=tmp_path / "requests.db",
        image_retention_mode="keep",
        cleanup_interval_seconds=9999,
    )
    app = create_app(settings)
    client = TestClient(app)

    image = _png()
    resp = client.post("/v1/analyze-image", files={"file": ("img.png", image, "image/png")})

    assert resp.status_code == 200
    row = app.state.repository.fetch_request(resp.json()["request_id"])
    assert row is not None
    assert row["retention_mode"] == "keep"
    assert row["expires_at"] == "9999-12-31T23:59:59Z"


def test_analyze_image_filters_unsupported_detections(tmp_path: Path) -> None:
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

    image = _png()
    resp = client.post("/v1/analyze-image", files={"file": ("img.png", image, "image/png")})

    assert resp.status_code == 200
    body = resp.json()
    assert [item["class_name"] for item in body["detections"]] == ["battery"]
    assert [item["class_name"] for item in body["guidance"]] == ["battery"]


def test_analyze_image_returns_all_supported_object_instances(tmp_path: Path) -> None:
    settings = replace(_settings(tmp_path), max_response_detections=3)
    app = create_app(settings)
    client = TestClient(app)

    app.state.detector.ready = True
    app.state.detector.model_version = "test-model"
    app.state.detector.predict = lambda _payload: [
        {
            "class_id": 0,
            "class_name": "battery",
            "confidence": 0.96,
            "bbox": {"x1": 10, "y1": 10, "x2": 40, "y2": 40},
        },
        {
            "class_id": 0,
            "class_name": "battery",
            "confidence": 0.95,
            "bbox": {"x1": 50, "y1": 50, "x2": 80, "y2": 80},
        },
        {
            "class_id": 1,
            "class_name": "cable",
            "confidence": 0.94,
            "bbox": {"x1": 15, "y1": 60, "x2": 90, "y2": 75},
        },
        {
            "class_id": 6,
            "class_name": "laptop",
            "confidence": 0.93,
            "bbox": {"x1": 5, "y1": 5, "x2": 95, "y2": 95},
        },
    ]

    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("img.png", _png(width=100, height=100), "image/png")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [item["class_name"] for item in body["detections"]] == [
        "battery",
        "battery",
        "cable",
    ]
    assert [item["class_name"] for item in body["guidance"]] == ["battery", "cable"]


def test_totem_gate_keeps_raw_predictions_in_the_sidecar(tmp_path: Path) -> None:
    settings = replace(
        _settings(tmp_path),
        dominant_object_gate_enabled=True,
        dominant_object_min_area_ratio=0.20,
    )
    app = create_app(settings)
    client = TestClient(app)

    app.state.detector.ready = True
    app.state.detector.model_version = "test-model"
    app.state.detector.predict = lambda _payload: [
        {
            "class_id": 3,
            "class_name": "mobile_phone_tablet",
            "confidence": 0.96,
            "bbox": {"x1": 10, "y1": 10, "x2": 90, "y2": 90},
        },
        {
            "class_id": 1,
            "class_name": "cable",
            "confidence": 0.80,
            "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        },
    ]

    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("img.png", _png(width=100, height=100), "image/png")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [item["class_name"] for item in body["detections"]] == ["mobile_phone_tablet"]
    sidecar = (settings.uploads_dir / f"{body['request_id']}.txt").read_text(encoding="utf-8")
    assert sidecar.splitlines() == [
        "3 0.500000 0.500000 0.800000 0.800000",
        "1 0.050000 0.050000 0.100000 0.100000",
    ]


def test_analyze_image_reports_exif_transposed_dimensions(tmp_path: Path) -> None:
    output = io.BytesIO()
    image = Image.new("RGB", (40, 20), color=(255, 255, 255))
    exif = Image.Exif()
    exif[274] = 6
    image.save(output, format="JPEG", exif=exif)

    app = create_app(_settings(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("rotated.jpg", output.getvalue(), "image/jpeg")},
    )

    assert resp.status_code == 200
    assert resp.json()["image_width"] == 20
    assert resp.json()["image_height"] == 40
