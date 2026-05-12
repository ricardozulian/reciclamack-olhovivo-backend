# Backend API

## Environment variables
- `MODEL_PATH` (default: `backend/app/model/yolo11n_ewaste.onnx`)
- `HAZARDS_PATH` (default: `backend/app/data/hazards_rules.json`)
- `COLLECTION_POINTS_PATH`
- `UPLOADS_DIR`
- `SQLITE_PATH`
- `IMAGE_RETENTION_HOURS` (default `24`)
- `CLEANUP_INTERVAL_SECONDS` (default `3600`)
- `MIN_CONFIDENCE` (default `0.40`)
- `NMS_IOU` (default `0.45`)
- `INPUT_SIZE` (default `640`)
- `MAX_UPLOAD_MB` (default `10`)
- `CORS_ALLOW_ORIGINS` (comma-separated origins, example: `http://<EC2_IP>:8080,https://<tunnel-host>`)

## Dataset Version Note
- `hazards_rules.json` is the current v1 compatibility content file.
- `hazards_rules_v2.json` will be used for v2 runs.
- For v2, start backend with `HAZARDS_PATH=backend/app/data/hazards_rules_v2.json`.

## API response contract (`POST /v1/analyze-image`)
- `request_id`
- `model_version`
- `content_version`
- `processed_at`
- `detections[]`: `class_id`, `class_name`, `confidence`, `bbox{x1,y1,x2,y2}`
- `guidance[]`: `class_name`, `typical_contents[]`, `hazard_summary`, `disposal_steps[]`, `legal_basis[]`
- `uncertainty_flag`
- `next_best_action`

## Testing
- Run: `python -m pytest`

Current tests cover:
- health/classes endpoints
- file-type validation
- analyze-image response shape
