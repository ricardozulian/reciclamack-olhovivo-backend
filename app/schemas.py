from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    display_label_pt_br: str
    confidence: float
    bbox: BoundingBox


class GuidanceItem(BaseModel):
    class_name: str
    display_label_pt_br: str
    typical_contents: list[str]
    hazard_summary: str
    disposal_steps: list[str]
    legal_basis: list[str]


class AnalyzeImageResponse(BaseModel):
    request_id: str
    model_version: str
    content_version: str
    processed_at: datetime
    detections: list[Detection]
    guidance: list[GuidanceItem]
    uncertainty_flag: bool
    next_best_action: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_version: str
    content_version: str


class ClassHint(BaseModel):
    class_name: str
    display_label_pt_br: str
    confidence_hint: str = Field(
        description="Human-friendly confidence interpretation for frontend display."
    )


class ClassesResponse(BaseModel):
    content_version: str
    classes: list[ClassHint]
