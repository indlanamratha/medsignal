# MedSignal prediction API.
# Build:  docker build -t medsignal-api .
# Run:    docker run -p 8000:8000 medsignal-api
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv

WORKDIR /app

# Install only what the API needs, pinned to the exact versions in uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o /tmp/constraints.txt \
    && uv pip install --system --no-cache -c /tmp/constraints.txt \
       fastapi uvicorn pydantic pandas numpy scikit-learn lightgbm joblib duckdb

COPY src ./src
RUN uv pip install --system --no-cache --no-deps .

COPY models/serious_model.joblib ./models/serious_model.joblib

RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs
USER appuser

ENV MODEL_PATH=/app/models/serious_model.joblib \
    PREDICTION_LOG=/app/logs/predictions.jsonl
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "medsignal.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
