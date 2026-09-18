FROM python:3.12.6-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12.6-slim

LABEL org.opencontainers.image.title="Async Research Assistant" \
      org.opencontainers.image.version="1.0"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY ai ./ai
COPY src ./src
COPY data ./data
COPY researcher.py requirements.txt ./

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "researcher", "demo", "--offline"]