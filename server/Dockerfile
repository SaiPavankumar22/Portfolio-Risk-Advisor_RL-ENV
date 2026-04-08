# Portfolio Risk Advisor — Docker Image
# ======================================
# Multi-stage build compatible with both:
#   - Standalone (openenv-core from PyPI)
#   - In-repo (openenv vendored via build context)
#
# Build:
#   docker build -t portfolio-risk-env -f server/Dockerfile .
#
# Run:
#   docker run -p 7860:7860 -e HF_TOKEN=<token> portfolio-risk-env

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY server/requirements.txt /app/requirements.txt

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ---- Final stage ----
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Ensure stdout/stderr are unbuffered (critical for log capture in Docker)
ENV PYTHONUNBUFFERED=1

# Expose HF Spaces standard port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
