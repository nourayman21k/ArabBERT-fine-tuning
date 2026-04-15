# 🌟 AraBERT Arabic Sentiment Analysis — Fine-tuned & Deployed

A complete end-to-end Arabic NLP project: fine-tuning `aubmindlab/bert-base-arabertv02` on 250,000 Arabic hotel reviews for 3-class sentiment classification, then packaging the result into a production-ready REST API served via Docker.

---

## 📊 Model Performance

| Class | Label | Per-class F1 |
|-------|-------|-------------|
| 0 | Negative 😠 | 83.5% |
| 1 | Neutral 😐 | 75.4% |
| 2 | Positive 😊 | 94.3% |

**Overall Accuracy: 88.86% · Macro F1: 89.1%**

---

## 🏗️ Project Structure

```
arabert-deployment/
├── ARabert_sentiment_analysis.ipynb   ← Full training notebook (Kaggle / Colab)
├── app/
│   ├── __init__.py
│   └── main.py                        ← FastAPI inference server
├── model/                             ← Place your model files here
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── static/
│   └── index.html                     ← Interactive Arabic web UI
├── Dockerfile                         ← CPU image (multi-stage build)
├── Dockerfile.gpu                     ← GPU image (NVIDIA CUDA)
├── docker-compose.yml
├── requirements.txt                   ← CPU dependencies
├── requirements-gpu.txt               ← GPU dependencies
└── README.md
```

---

## 🔬 Training Pipeline (Notebook)

The training notebook (`ARabert_sentiment_analysis.ipynb`) covers the full ML lifecycle:

**1. Data — Arabic Hotel Reviews**
- Source: `nourayman21k/arabic-dataset-hotels` on Kaggle
- Raw dataset loaded from a tab-separated UTF-16-LE file
- Cleaned: dropped nulls, removed very short texts, deduplicated
- Rating → label mapping: `{1,2} → negative`, `{3} → neutral`, `{4,5} → positive`
- Sampled 250,000 reviews for training efficiency

**2. Data Splits**
- 80/10/10 train/val/test split using stratified sampling to preserve class balance

**3. Tokenization**
- `aubmindlab/bert-base-arabertv02` tokenizer, max length 128 tokens
- Custom PyTorch `Dataset` class returning `input_ids`, `attention_mask`, `token_type_ids`, and `labels`

**4. Class Imbalance Handling**
- `sklearn.utils.class_weight.compute_class_weight` to compute balanced class weights
- Custom `WeightedTrainer` subclassing HuggingFace `Trainer`, overriding `compute_loss` to apply weighted `CrossEntropyLoss`

**5. Model**
- `BertForSequenceClassification` from `aubmindlab/bert-base-arabertv02` with `num_labels=3`
- Trained on GPU (CUDA) when available, CPU fallback

**6. Training Configuration**
| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| FP16 | Yes (GPU) |
| Best model metric | Weighted F1 |

**7. Evaluation**
- Weighted F1, per-class F1 (negative / neutral / positive), and accuracy tracked each epoch

**8. Saving**
- Model weights + config saved via `trainer.model.save_pretrained()`
- Tokenizer saved via `tokenizer.save_pretrained()`
- Output zipped for easy download from Kaggle

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed
- (GPU only) NVIDIA Container Toolkit

### Step 1 — Add your model files

```bash
mkdir -p model
cp /path/to/model.safetensors   ./model/
cp /path/to/config.json         ./model/
cp /path/to/tokenizer.json      ./model/
cp /path/to/tokenizer_config.json ./model/
```

### Step 2 — Run (CPU)

```bash
docker compose up --build
# API → http://localhost:8000
# Docs → http://localhost:8000/docs
```

### Step 2 (alt) — Run (GPU)

```bash
docker compose --profile gpu up --build
# API → http://localhost:8001
```

---

## 🌐 API Reference

### `POST /predict` — Single text

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "المنتج رائع جداً وأنصح به الجميع"}'
```

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

### `POST /predict/batch` — Up to 32 texts

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["المنتج رائع", "الخدمة سيئة جداً", "لا بأس به"]}'
```

### `GET /health`

```json
{"status": "ready", "device": "cpu"}
```

### Interactive Swagger Docs

Open **http://localhost:8000/docs** for the full Swagger UI.

---

## 🖥️ Web UI

A fully RTL Arabic web interface (`index.html`) is bundled and served at `http://localhost:8000`:

- Enter any Arabic text and click **Analyze**
- Instant probability bar chart across all three classes (positive / neutral / negative)
- Preset example chips to get started quickly
- Recent analysis history panel
- Dark mode support
- Fully responsive

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/app/model` | Path to model files |
| `MAX_LENGTH` | `512` | Max token length |
| `PORT` | `8000` | Server port |

---

## 🐳 Docker Details

**CPU image** (`Dockerfile`):
- Multi-stage build: `python:3.11-slim` builder → slim runtime
- Non-root user (`appuser`) for security
- Health check on `/health` every 30s
- `libgomp1` for PyTorch OpenMP threading

**GPU image** (`Dockerfile.gpu`):
- Same structure, served on port `8001` via Docker Compose
- Requires NVIDIA Container Toolkit on the host
- Switch to `torch` with CUDA support by using `requirements-gpu.txt`

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | `aubmindlab/bert-base-arabertv02` |
| Fine-tuning | HuggingFace Transformers + PyTorch |
| API | FastAPI + Uvicorn |
| Containerization | Docker (multi-stage), Docker Compose |
| Schema validation | Pydantic v2 |
| Frontend | Vanilla HTML/CSS/JS (RTL Arabic) |
| Training platform | Kaggle (GPU T4) |

---

## 📝 Notes

- The model is mounted as a **read-only Docker volume** by default, keeping the image small and allowing model hot-swaps without rebuilding.
- For production: add Nginx/Traefik in front for TLS, and enable Gunicorn with multiple workers if running on a multi-core host (use separate processes, not threads, due to PyTorch's GIL behaviour).
- The tokenizer's `never_split` list includes Arabic-specific special tokens: `[بريد]`, `[مستخدم]`, `[رابط]` (email, user, URL).

---

## 🙏 Credits

- Base model: [aubmindlab/bert-base-arabertv02](https://huggingface.co/aubmindlab/bert-base-arabertv02)
- Dataset: [nourayman21k — Arabic Hotel Reviews](https://www.kaggle.com/datasets/nourayman21k/arabic-dataset-hotels)
