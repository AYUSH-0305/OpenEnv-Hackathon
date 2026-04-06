# Root-level Dockerfile for HuggingFace Spaces deployment
# Build context is the repo root — copies the full med_reconciliation package

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy the full med_reconciliation package
COPY med_reconciliation/ /app/med_reconciliation/

# Install dependencies
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "pydantic>=2.0.0"

# Set PYTHONPATH so imports resolve correctly
ENV PYTHONPATH=/app
ENV MED_RECON_TASK=hard

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# HuggingFace Spaces uses port 7860
CMD ["uvicorn", "med_reconciliation.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
