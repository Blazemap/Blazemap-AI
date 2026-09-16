FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=0
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-cache

FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
COPY --from=build /app/.venv ./.venv
COPY main.py schemas.py ./
USER 10001:10001
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn main:app --host '' --port \"${PORT:-8000}\" --workers 1 --loop asyncio"]
