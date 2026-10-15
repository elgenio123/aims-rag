FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY config.py main.py ./
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN useradd --uid 1000 --create-home app \
    && mkdir -p /app/logs /app/data/documents \
    && chown -R app:app /app/logs /app/data \
    && chmod 1777 /app/logs

FROM base AS test
RUN uv sync --frozen --no-install-project
COPY tests/ ./tests/
USER app
CMD ["python", "-m", "pytest", "-v"]

FROM base AS runtime
USER app
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
