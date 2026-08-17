# Multi-stage production Dockerfile for ProductStudio AI
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and requirements
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130 && \
    pip install --no-cache-dir -e ".[tracking]"

# --- Production Image ---
FROM python:3.11-slim AS final

WORKDIR /app

# Install runtime system libraries for OpenCV/PIL and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 10001 appuser && \
    mkdir -p /app/outputs /app/mlruns /app/data && \
    chown -R appuser:appuser /app

# Copy python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser data/ /app/data/
COPY --chown=appuser:appuser eval/ /app/eval/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser pyproject.toml README.md /app/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PRODUCTSTUDIO_OUTPUT_DIR=/app/outputs \
    PRODUCTSTUDIO_DATABASE_URL=sqlite:////app/outputs/productstudio.db \
    PRODUCTSTUDIO_MLFLOW_TRACKING_URI=/app/mlruns

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "app.main", "--host", "0.0.0.0", "--port", "8000"]
