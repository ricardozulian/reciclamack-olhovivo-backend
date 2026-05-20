# ReciclaMack Olho Vivo — Backend

API do projeto de extensão universitária **Olho Vivo — Identificação de Resíduos Eletroeletrônicos por Visão Computacional**, desenvolvido no âmbito da Universidade Presbiteriana Mackenzie, Faculdade de Computação e Informática (FCI).

O backend recebe imagens enviadas pelo frontend, executa inferência com modelo YOLO11 exportado para ONNX e retorna detecções, orientações ambientais e pontos de coleta relacionados ao descarte correto de resíduos eletroeletrônicos.

## Contexto acadêmico

- Instituição: Universidade Presbiteriana Mackenzie
- Unidade: Faculdade de Computação e Informática (FCI)
- Área temática: Meio Ambiente, Tecnologia e Produção, Educação Ambiental
- Linha de extensão: Gestão de Resíduos Sólidos e Educação para a Sustentabilidade
- Coordenação/orientação: Profa. Sandra Bozolan

## Equipe discente

- Ricardo Zulian de Souza Amaral
- Marcos Volponi Cervan
- Flavio Estevam Nogueira Andrade

## Escopo técnico

- API REST em Python com FastAPI.
- Inferência local em CPU usando ONNX Runtime.
- Modelo runtime padrão de teste v2 em `app/model/yolo11s_ewaste_v2_25class_512.onnx`.
- Conteúdo ambiental em JSON.
- Registro operacional leve em SQLite.
- Retenção configurável de imagens enviadas, com rótulos `.txt` pareados em formato YOLO para depuração e evolução do modelo.

## Variáveis de ambiente

- `MODEL_PATH`: caminho do modelo ONNX. Para teste v2: `backend/app/model/yolo11s_ewaste_v2_25class_512.onnx`.
- `MODEL_CLASSES_PATH`: ordem de classes do modelo. Para v2 512: `backend/app/model/ewaste_v2_25class.classes.txt`.
- `HAZARDS_PATH`: conteúdo ambiental. Para v2 25 classes: `backend/app/data/hazards_rules_v2_25class.json`.
- `COLLECTION_POINTS_PATH`: base de pontos de coleta.
- `UPLOADS_DIR`: diretório temporário de uploads.
- `SQLITE_PATH`: banco SQLite operacional.
- `IMAGE_RETENTION_MODE`: política de retenção. `ttl` apaga após o prazo; `keep` preserva as imagens e os `.txt` pareados para depuração e treinamento. Padrão: `ttl`.
- `IMAGE_RETENTION_HOURS`: retenção de imagens quando `IMAGE_RETENTION_MODE=ttl`. Padrão: `24`.
- `CLEANUP_INTERVAL_SECONDS`: intervalo da limpeza. Padrão: `3600`.
- `MIN_CONFIDENCE`: confiança mínima. Padrão: `0.40`.
- `NMS_IOU`: limiar de NMS. Padrão: `0.45`.
- `INPUT_SIZE`: tamanho de entrada do modelo. Para v2 512: `512`.
- `MAX_UPLOAD_MB`: tamanho máximo de upload. Padrão: `10`.
- `CORS_ALLOW_ORIGINS`: origens permitidas, separadas por vírgula.
- `RATE_LIMIT_ANALYZE_PER_MINUTE`: limite por IP para `POST /v1/analyze-image`. Padrão: `30`.
- `ENABLE_API_DOCS`: habilita `/docs`, `/redoc` e `/openapi.json` quando definido como `1`, `true`, `yes` ou `on`. Padrão: desabilitado.

## Contrato principal da API

Endpoint:

```text
POST /v1/analyze-image
```

Campos principais da resposta:

- `request_id`
- `model_version`
- `content_version`
- `processed_at`
- `detections[]`: `class_id`, `class_name`, `confidence`, `bbox`
- `guidance[]`: `class_name`, `typical_contents`, `hazard_summary`, `disposal_steps`, `legal_basis`
- `uncertainty_flag`
- `next_best_action`

## Executar localmente

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Testes

```powershell
python -m pytest
```

## Papel no sistema

Este repositório é autônomo e contém apenas a API de inferência e conteúdo ambiental. O frontend e o pipeline de treinamento ficam em repositórios separados.
