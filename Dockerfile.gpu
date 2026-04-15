# ── Stage 1: builder — install Python deps ─────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install/deps --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="aymannour041@gmail.com"
LABEL description="AraBERT Arabic Sentiment Analysis API"

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local

# Copy application code
COPY . . 

# Copy model weights  ← you will mount or COPY your model directory here
# Option A (bake into image — larger image, simpler deploy):
#   COPY model/ ./model/
# Option B (mount at runtime via Docker volume — recommended for large models):
#   docker run -v /path/to/your/model:/app/model ...
# The ENV below sets the default path; override with -e MODEL_DIR=...
ENV MODEL_DIR=/app/model \
    MAX_LENGTH=512 \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# HuggingFace cache (avoids permission issues)
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]
