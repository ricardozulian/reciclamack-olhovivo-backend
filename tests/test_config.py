from __future__ import annotations

from app.config import get_settings


def test_get_settings_reads_explicit_model_version(monkeypatch) -> None:
    monkeypatch.setenv(
        "MODEL_VERSION",
        "v2_1_2_letterbox_enhanced_adamw_e0_512_yolo11s_epoch50",
    )

    settings = get_settings()

    assert settings.model_version == (
        "v2_1_2_letterbox_enhanced_adamw_e0_512_yolo11s_epoch50"
    )
