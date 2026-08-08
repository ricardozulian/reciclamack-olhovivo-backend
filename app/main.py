from __future__ import annotations

import asyncio
import io
import ipaddress
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .cleanup import cleanup_loop
from .config import Settings, get_settings
from .content import ContentStore, normalize_class_name
from .inference import DetectorConfig, OnnxDetector
from .repository import RequestRepository
from .schemas import AnalyzeImageResponse, ClassHint, ClassesResponse, GuidanceItem, HealthResponse
from .storage import UploadStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reciclamack.api")

MODEL_CLASS_ALIASES = {
    "bateria": "battery",
    "cabo": "cable",
    "celular": "mobile_phone_tablet",
    "impressora": "printer_multifunction",
    "monitor": "flat_monitor",
    "notebook": "laptop",
    "placa_eletronica": "computer_part",
    "player": "portable_music_player",
    "router": "network_device",
    "telephone": "landline_telephone",
    "televisao": "crt_monitor",
}


def _client_ip(request: Request) -> str:
    cloudfront_viewer = request.headers.get("cloudfront-viewer-address", "").strip()
    if cloudfront_viewer:
        host = cloudfront_viewer.rsplit(":", 1)[0].strip("[]")
        if host:
            return host

    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        candidates = [part.strip() for part in forwarded.split(",") if part.strip()]
        for candidate in reversed(candidates):
            try:
                ip = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            if not ip.is_private and not ip.is_loopback and not ip.is_link_local:
                return candidate
        if candidates:
            return candidates[-1]
    return request.client.host if request.client else "unknown"


def _confidence_hint(confidence: float) -> str:
    if confidence >= 0.85:
        return "Alta confiança"
    if confidence >= 0.60:
        return "Confiança moderada"
    return "Confiança baixa"


def _load_model_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        names.append(normalize_class_name(MODEL_CLASS_ALIASES.get(name.lower(), name)))
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


def _active_content_classes(model_class_names: list[str], content_store: ContentStore) -> list[str]:
    content_set = set(content_store.supported_classes())
    active: list[str] = []
    seen: set[str] = set()
    for name in model_class_names:
        class_name = name.lower()
        if class_name in content_set and class_name not in seen:
            active.append(class_name)
            seen.add(class_name)
    return active or content_store.supported_classes()


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _image_size(payload: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return ImageOps.exif_transpose(image).size
    except (OSError, UnidentifiedImageError):
        return None


def _verify_image_payload(payload: bytes) -> None:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem válida.") from exc


def _to_yolo_label_line(detection: dict[str, object], image_size: tuple[int, int]) -> str | None:
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    bbox = detection.get("bbox")
    if not isinstance(bbox, dict):
        return None

    x1 = _clamp(float(bbox["x1"]) / width)
    y1 = _clamp(float(bbox["y1"]) / height)
    x2 = _clamp(float(bbox["x2"]) / width)
    y2 = _clamp(float(bbox["y2"]) / height)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    x_center = (left + right) / 2
    y_center = (top + bottom) / 2
    box_width = right - left
    box_height = bottom - top
    return (
        f"{int(detection['class_id'])} "
        f"{x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
    )


def _dataset_label_content(
    detections: list[dict[str, object]],
    image_size: tuple[int, int] | None,
    inference_failed: bool,
) -> str:
    if inference_failed or not detections or image_size is None:
        return "null"
    lines = [
        line
        for line in (_to_yolo_label_line(detection, image_size) for detection in detections)
        if line is not None
    ]
    return "\n".join(lines) if lines else "null"


def _filter_v1_detections(
    detections: list[dict[str, object]],
    supported_classes: set[str],
    min_confidence: float,
    max_response_detections: int,
) -> list[dict[str, object]]:
    eligible = [
        detection
        for detection in detections
        if float(detection.get("confidence", 0.0)) >= min_confidence
        and str(detection.get("class_name", "")).lower() in supported_classes
    ]
    eligible.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    if max_response_detections > 0:
        return eligible[:max_response_detections]
    return eligible


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app = FastAPI(
        title=app_settings.app_name,
        version="0.2.0",
        docs_url="/docs" if app_settings.enable_api_docs else None,
        redoc_url="/redoc" if app_settings.enable_api_docs else None,
        openapi_url="/openapi.json" if app_settings.enable_api_docs else None,
    )
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
    active_class_names = _active_content_classes(model_class_names, content_store)

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
    app.state.detector = detector
    app.state.content_store = content_store
    app.state.active_class_names = active_class_names

    repository = RequestRepository(app_settings.sqlite_path)
    repository.init()
    app.state.repository = repository
    app.state.settings = app_settings
    storage = UploadStorage(
        app_settings.uploads_dir,
        app_settings.image_retention_hours,
        app_settings.image_retention_mode,
    )
    cleanup_task: asyncio.Task | None = None
    analyze_hits: dict[str, deque[float]] = defaultdict(deque)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=()")
        return response

    class LoggedRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
            original_route_handler = super().get_route_handler()

            async def custom_route_handler(request: Request) -> Response:
                started = time.perf_counter()
                req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
                limit = app_settings.rate_limit_analyze_per_minute
                if limit > 0 and request.method == "POST" and request.url.path == "/v1/analyze-image":
                    now = time.monotonic()
                    window_start = now - 60
                    key = _client_ip(request)
                    hits = analyze_hits[key]
                    while hits and hits[0] < window_start:
                        hits.popleft()
                    if len(hits) >= limit:
                        elapsed_ms = int((time.perf_counter() - started) * 1000)
                        logger.warning(
                            "rate_limited request_id=%s client_ip=%s path=%s limit=%s latency_ms=%s",
                            req_id,
                            key,
                            request.url.path,
                            limit,
                            elapsed_ms,
                        )
                        response = JSONResponse(
                            {"detail": "Muitas requisições. Tente novamente em instantes."},
                            status_code=429,
                        )
                        response.headers["X-Request-Id"] = req_id
                        return response
                    hits.append(now)
                try:
                    response = await original_route_handler(request)
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

            return custom_route_handler

    app.router.route_class = LoggedRoute

    @app.on_event("startup")
    async def on_startup() -> None:
        nonlocal cleanup_task
        cleanup_task = asyncio.create_task(
            cleanup_loop(repository, app_settings.cleanup_interval_seconds)
        )
        logger.info("Cleanup loop started")

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
                for name in active_class_names
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
        _verify_image_payload(payload)

        request_id = str(uuid.uuid4())
        ext = Path(file.filename or "").suffix.lower() or ".jpg"
        stored_path, expires_at = storage.save(request_id, payload, ext)

        started = time.perf_counter()
        inference_failed = False
        try:
            detections = detector.predict(payload)
        except Exception:
            logger.exception("Inference failed for request_id=%s", request_id)
            detections = []
            inference_failed = True
        inference_ms = int((time.perf_counter() - started) * 1000)

        filtered = _filter_v1_detections(
            detections=detections,
            supported_classes=set(active_class_names),
            min_confidence=app_settings.min_confidence,
            max_response_detections=app_settings.max_response_detections,
        )
        uncertainty_flag = len(filtered) == 0
        next_best_action = (
            "Não foi possível identificar com boa confiança. Tente outra foto com melhor iluminação."
            if uncertainty_flag
            else "Confira as orientações e entregue o material em ponto de coleta autorizado."
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

        image_size = _image_size(payload)
        label_content = _dataset_label_content(
            response_detections,
            image_size,
            inference_failed,
        )
        try:
            storage.save_label(stored_path, label_content)
        except Exception:
            logger.exception("Failed to persist dataset label for request_id=%s", request_id)

        try:
            repository.insert_request(
                request_id=request_id,
                stored_path=stored_path.as_posix(),
                expires_at=expires_at,
                retention_mode=app_settings.image_retention_mode,
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
            image_width=image_size[0] if image_size else 0,
            image_height=image_size[1] if image_size else 0,
            detections=response_detections,
            guidance=guidance,
            uncertainty_flag=uncertainty_flag,
            next_best_action=next_best_action,
        )

    return app


app = create_app()
