# ReciclaMack Olho Vivo Backend

This repository contains the API for the ReciclaMack university extension project.

The backend receives images, runs YOLO11 ONNX inference, and returns detections and disposal guidance.

## Academic context

- Institution: Universidade Presbiteriana Mackenzie
- School: Faculdade de Computação e Informática (FCI)
- Coordinator: Professor Sandra Bozolan

## Student team

- Ricardo Zulian de Souza Amaral
- Marcos Volponi Cervan
- Flavio Estevam Nogueira Andrade

## Technical scope

- Python REST API with FastAPI.
- CPU inference with ONNX Runtime.
- Default test model: `app/model/yolo11s_ewaste_v2_25class_512.onnx`.
- Environmental guidance in JSON.
- Operational metadata in SQLite.
- Configurable image retention with matching YOLO sidecar files.

## Inference geometry

The current backend uses centered letterbox preprocessing. It preserves the
oriented source aspect ratio and uses `(114, 114, 114)` padding.

The backend removes the padding from output coordinates. It divides coordinates
by the letterbox scale and clamps each box to the source image.

The active comparison settings are confidence `0.40`, NMS IoU `0.45`, and a
maximum of eight detections. Direct `512 x 512` resize is a legacy policy.

Read `../documentation/inference_preprocessing_policy.md` before a model or
inference change.

## Environment variables

- `MODEL_PATH`: ONNX model path.
- `MODEL_CLASSES_PATH`: class file path and output order.
- `HAZARDS_PATH`: environmental guidance file.
- `COLLECTION_POINTS_PATH`: collection point database.
- `UPLOADS_DIR`: temporary upload directory.
- `SQLITE_PATH`: operational SQLite database.
- `IMAGE_RETENTION_MODE`: `ttl` removes expired files. `keep` retains image and sidecar pairs.
- `IMAGE_RETENTION_HOURS`: image lifetime in `ttl` mode. Default: `24`.
- `CLEANUP_INTERVAL_SECONDS`: cleanup interval. Default: `3600`.
- `MIN_CONFIDENCE`: minimum confidence. Default: `0.40`.
- `NMS_IOU`: NMS threshold. Default: `0.45`.
- `INPUT_SIZE`: model input size. The v2 model uses `512`.
- `MAX_UPLOAD_MB`: maximum upload size. Default: `10`.
- `CORS_ALLOW_ORIGINS`: comma-separated allowed origins.
- `RATE_LIMIT_ANALYZE_PER_MINUTE`: per-IP limit for `POST /v1/analyze-image`. Default: `30`.
- `MAX_RESPONSE_DETECTIONS`: maximum detections in one response. Default: `8`.
- `ENABLE_API_DOCS`: enables `/docs`, `/redoc`, and `/openapi.json`. Default: off.

## Main API contract

```text
POST /v1/analyze-image
```

The main response fields are:

- `request_id`
- `model_version`
- `content_version`
- `processed_at`
- `image_width`
- `image_height`
- `detections[]`
- `guidance[]`
- `uncertainty_flag`
- `next_best_action`

Detection boxes use pixel coordinates from the oriented display image.

The frontend draws the boxes on the oriented display image.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run tests

```powershell
python -m pytest
```

## Docker integration environment

Run the Docker test host from the workspace root:

```powershell
.\deploy\local\run_model.ps1
```

The local Compose file sets a seven-day TTL and a one-day cleanup interval.

It enables API documentation and disables rate limits only for LAN tests.

The runtime normalizes `home_theater` to `av_equipment` and keeps class ID 24.

Read `../deploy/MODEL_TEST_HANDOFF.md` before a model change.

## Totem behavior

The totem preview stays in Chromium on the Jetson.

Each user action sends one image to the local API. The system does not send continuous camera video.

Stored predictions and sidecars are unaudited automatic results.
