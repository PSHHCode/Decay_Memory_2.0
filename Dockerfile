# =============================================================================
# Decay Memory Backend - Optimized Dockerfile (Multi-Stage)
# Fixes OOM crashes on low-memory VPS during pip compilation
# =============================================================================

# --- STAGE 1: Builder (compiles any C extensions) ---
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies (only in builder stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for clean isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with binary preference
# Note: plyer removed (desktop notifications don't work in containers)
RUN pip install --no-cache-dir --prefer-binary \
    fastapi \
    uvicorn[standard] \
    pydantic \
    python-dotenv \
    google-generativeai \
    openai \
    anthropic \
    groq \
    deepgram-sdk \
    elevenlabs \
    qdrant-client \
    networkx \
    colorama \
    python-multipart \
    requests

# --- STAGE 2: Runtime (slim, no build tools) ---
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy only the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY *.py ./
COPY *.json ./

# Create directories for runtime data
RUN mkdir -p /app/flight_recorders

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (internal, nginx will proxy to this)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
