from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .cleanup import cleanup_loop
from .config import Settings, get_settings
from .content import ContentStore
from .inference import DetectorConfig, OnnxDetector
from .repository import RequestRepository
from .schemas import AnalyzeImageResponse, ClassHint, ClassesResponse, GuidanceItem, HealthResponse
from .storage import UploadStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reciclamack.api")


def _confidence_hint(confidence: float) -> str:
    if confidence >= 0.85:
        return "Alta confianca"
    if confidence >= 0.60:
        return "Confianca moderada"
    return "Confianca baixa"


def _load_model_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        names.append(name)
    return names


def _validate_class_parity(model_class_names: list[str], content_store: ContentStore) -> None:
    model_set = set(model_class_names)
    content_set = set(content_store.supported_classes())
    missing_in_content = sorted(model_set - content_set)
    missing_in_model = sorted(content_set - model_set)
    if missing_in_content or missing_in_model:
        logger.warning(
            "Class parity mismatch between model classes and hazards content. "
            "Continuing in compatibility mode for demo/runtime. "
            "missing_in_content=%s missing_in_model=%s",
            missing_in_content,
            missing_in_model,
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app = FastAPI(title=app_settings.app_name, version="0.1.0")
    if app_settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(app_settings.cors_allow_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    content_store = ContentStore(app_settings.hazards_path)
    content_store.load()
    model_class_names = _load_model_class_names(app_settings.model_classes_path)
    if not model_class_names:
        model_class_names = content_store.supported_classes()
        logger.warning(
            "Model class list file not found at %s; falling back to content classes.",
            app_settings.model_classes_path,
        )
    _validate_class_parity(model_class_names, content_store)

    detector = OnnxDetector(
        DetectorConfig(
            model_path=app_settings.model_path,
            input_size=app_settings.input_size,
            confidence_threshold=app_settings.min_confidence,
            nms_iou=app_settings.nms_iou,
            class_names=model_class_names,
        )
    )
    detector.load()

    repository = RequestRepository(app_settings.sqlite_path)
    repository.init()
    storage = UploadStorage(app_settings.uploads_dir, app_settings.image_retention_hours)
    cleanup_task: asyncio.Task | None = None

    @app.on_event("startup")
    async def on_startup() -> None:
        nonlocal cleanup_task
        cleanup_task = asyncio.create_task(
            cleanup_loop(repository, app_settings.cleanup_interval_seconds)
        )
        logger.info("Cleanup loop started")

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        started = time.perf_counter()
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.exception(
                "request_failed request_id=%s method=%s path=%s latency_ms=%s",
                req_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Request-Id"] = req_id
        logger.info(
            "request_done request_id=%s method=%s path=%s status=%s latency_ms=%s",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_ready=detector.ready,
            model_version=detector.model_version,
            content_version=content_store.content_version,
        )

    @app.get("/v1/classes", response_model=ClassesResponse)
    def classes() -> ClassesResponse:
        return ClassesResponse(
            content_version=content_store.content_version,
            classes=[
                ClassHint(
                    class_name=name,
                    display_label_pt_br=content_store.get_display_label(name),
                    confidence_hint=_confidence_hint(app_settings.min_confidence),
                )
                for name in content_store.supported_classes()
            ],
        )

    @app.post("/v1/analyze-image", response_model=AnalyzeImageResponse)
    async def analyze_image(file: UploadFile = File(...)) -> AnalyzeImageResponse:
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem.")

        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Imagem vazia.")
        if len(payload) > app_settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Imagem acima do limite permitido.")

        request_id = str(uuid.uuid4())
        ext = Path(file.filename or "").suffix.lower() or ".jpg"
        stored_path, expires_at = storage.save(request_id, payload, ext)

        started = time.perf_counter()
        try:
            detections = detector.predict(payload)
        except Exception:
            logger.exception("Inference failed for request_id=%s", request_id)
            detections = []
        inference_ms = int((time.perf_counter() - started) * 1000)

        filtered = [d for d in detections if d["confidence"] >= app_settings.min_confidence]
        uncertainty_flag = len(filtered) == 0
        next_best_action = (
            "Nao foi possivel identificar com boa confianca. Tente outra foto com melhor iluminacao."
            if uncertainty_flag
            else "Confira as orientacoes e entregue o material em ponto de coleta autorizado."
        )

        seen: set[str] = set()
        guidance: list[GuidanceItem] = []
        response_detections: list[dict[str, object]] = []
        for detection in filtered:
            class_name = detection["class_name"]
            response_detections.append(
                {
                    **detection,
                    "display_label_pt_br": content_store.get_display_label(class_name),
                }
            )
            if class_name in seen:
                continue
            seen.add(class_name)
            info = content_store.get_guidance(class_name)
            if not info:
                continue
            guidance.append(
                GuidanceItem(
                    class_name=class_name,
                    display_label_pt_br=content_store.get_display_label(class_name),
                    typical_contents=info.get("typical_contents", []),
                    hazard_summary=f"{info['hazard_level']}: {info['health_risks']}",
                    disposal_steps=info["disposal_instructions_br"],
                    legal_basis=info["legal_references"],
                )
            )

        try:
            repository.insert_request(
                request_id=request_id,
                stored_path=stored_path.as_posix(),
                expires_at=expires_at,
                model_version=detector.model_version,
                inference_ms=inference_ms,
            )
        except Exception:
            logger.exception("Failed to persist request metadata for request_id=%s", request_id)

        return AnalyzeImageResponse(
            request_id=request_id,
            model_version=detector.model_version,
            content_version=content_store.content_version,
            processed_at=datetime.now(timezone.utc),
            detections=response_detections,
            guidance=guidance,
            uncertainty_flag=uncertainty_flag,
            next_best_action=next_best_action,
        )

    return app


app = create_app()
