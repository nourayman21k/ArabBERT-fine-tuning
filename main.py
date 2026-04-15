"""
AraBERT Sentiment Analysis — FastAPI Inference Server
Labels: 0 = Negative | 1 = Neutral | 2 = Positive
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, BertForSequenceClassification

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    0: {"label": "negative", "emoji": "😠"},
    1: {"label": "neutral",  "emoji": "😐"},
    2: {"label": "positive", "emoji": "😊"},
}

# ── Global model state ─────────────────────────────────────────────────────────
model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info(f"Loading AraBERT model from '{MODEL_DIR}' on {DEVICE} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    model_state["tokenizer"] = tokenizer
    model_state["model"] = model
    logger.info(f"Model ready in {time.time() - t0:.2f}s")
    yield
    model_state.clear()
    logger.info("Model unloaded.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AraBERT Sentiment API",
    description="Arabic sentiment analysis (Negative / Neutral / Positive) powered by fine-tuned AraBERT.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ─────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Schemas ────────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, example="المنتج رائع جداً وأنصح به الجميع")


class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=32, example=["المنتج رائع", "الخدمة سيئة"])


class SentimentResult(BaseModel):
    text: str
    label: str
    label_id: int
    emoji: str
    confidence: float
    scores: dict
    inference_ms: float


class BatchResult(BaseModel):
    results: List[SentimentResult]
    total_inference_ms: float


# ── Helpers ────────────────────────────────────────────────────────────────────
def run_inference(texts: List[str]) -> List[dict]:
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).cpu().tolist()
    preds = torch.argmax(logits, dim=-1).cpu().tolist()

    results = []
    for text, pred, prob in zip(texts, preds, probs):
        meta = LABEL_MAP[pred]
        results.append({
            "text": text,
            "label": meta["label"],
            "label_id": pred,
            "emoji": meta["emoji"],
            "confidence": round(prob[pred], 4),
            "scores": {
                "negative": round(prob[0], 4),
                "neutral":  round(prob[1], 4),
                "positive": round(prob[2], 4),
            },
        })
    return results


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Frontend"])
def serve_frontend():
    """Serve the HTML frontend UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "service": "AraBERT Sentiment API", "device": DEVICE}


@app.get("/health", tags=["Health"])
def health():
    ready = "model" in model_state
    return {"status": "ready" if ready else "loading", "device": DEVICE}


@app.post("/predict", response_model=SentimentResult, tags=["Inference"])
def predict(body: TextInput):
    """Predict sentiment for a single Arabic text."""
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    t0 = time.perf_counter()
    results = run_inference([body.text])
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return {**results[0], "inference_ms": elapsed}


@app.post("/predict/batch", response_model=BatchResult, tags=["Inference"])
def predict_batch(body: BatchInput):
    """Predict sentiment for a batch of Arabic texts (max 32)."""
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    t0 = time.perf_counter()
    results = run_inference(body.texts)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "results": [{**r, "inference_ms": elapsed / len(results)} for r in results],
        "total_inference_ms": elapsed,
    }
