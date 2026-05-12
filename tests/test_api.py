from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


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


def test_analyze_image_rejects_non_image(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    resp = client.post(
        "/v1/analyze-image",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


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
