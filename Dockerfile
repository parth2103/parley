FROM ghcr.io/astral-sh/uv:0.8.4 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev \
    && uv run --frozen --no-dev python -m nltk.downloader -d .venv/nltk_data punkt_tab
COPY backend ./backend
CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
