
https://github.com/user-attachments/assets/d721374e-6407-49aa-854a-d4555a1e4299


# AraBERT Sentiment Analysis — Deployment Guide

Fine-tuned AraBERT for 3-class Arabic sentiment classification.

| Label | Meaning | Eval F1 |
|-------|---------|---------|
| LABEL_0 | Negative 😠 | 83.5% |
| LABEL_1 | Neutral 😐 | 75.4% |
| LABEL_2 | Positive 😊 | 94.3% |

**Overall accuracy: 88.86% | Macro F1: 89.1%**

---

## Project Structure

```
arabert-deployment/
├── app/
│   └── main.py              ← FastAPI inference server
├── model/                   ← Place your model files here
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── Dockerfile               ← CPU image
├── Dockerfile.gpu           ← GPU image
├── docker-compose.yml
├── requirements.txt         ← CPU deps
├── requirements-gpu.txt     ← GPU deps
└── README.md
```

---

## Quick Start

### Step 1 — Place your model files
```bash
# Copy all 4 model files into the model/ folder:
cp /path/to/model.safetensors   ./model/
cp /path/to/config.json         ./model/
cp /path/to/tokenizer.json      ./model/
cp /path/to/tokenizer_config.json ./model/
```

### Step 2 — Build & run (CPU)
```bash
docker compose up --build
```

### Step 2 (alt) — GPU
```bash
docker compose --profile gpu up --build
# API will be on port 8001
```

---

## API Usage

### Single prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "المنتج رائع جداً وأنصح به الجميع"}'
```

Response:
```json
{
  "text": "المنتج رائع جداً وأنصح به الجميع",
  "label": "positive",
  "label_id": 2,
  "emoji": "😊",
  "confidence": 0.9823,
  "scores": {"negative": 0.0071, "neutral": 0.0106, "positive": 0.9823},
  "inference_ms": 48.3
}
```

### Batch prediction (up to 32 texts)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["المنتج رائع", "الخدمة سيئة جداً", "لا بأس به"]}'
```

### Interactive docs
Open **http://localhost:8000/docs** in your browser — full Swagger UI.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./model` | Path to model files |
| `MAX_LENGTH` | `512` | Max token length |
| `PORT` | `8000` | Server port |

---
