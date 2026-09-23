# Blazemap AI

The internal analysis service for Blazemap's forest and land fire awareness system in Kalimantan, Indonesia. Public website served by the separate frontend: [blazemap.my.id](https://blazemap.my.id/). Source code: [Blazemap-AI](https://github.com/Blazemap/Blazemap-AI).

## What it does

Receives a structured case context from the backend and returns source-referenced analysis for human review: evidence and impact levels, suggested priority, cited reasons, missing information, suggested checks, and limitations. It does **not** confirm fires, publish official warnings, dispatch teams, or order evacuations. The backend retains responsibility for authorization and operational decisions.

## Technology stack

| Area | Technology |
| --- | --- |
| Language and dependencies | Python >=3.12,<3.15; `uv` and `uv.lock` |
| HTTP API | FastAPI, Uvicorn, HTTPX |
| Schemas | Pydantic 2 |
| Model integration | Official Google Gen AI SDK and Gemini Developer API |
| Source checks | Ruff and mypy |

## Installation and local development

Prerequisites: a compatible Python version, `uv`, and credentials for the Gemini Developer API. From this repository:

```bash
uv sync --locked
```

Use `.env.example` as the list of required service settings:

```dotenv
AI_SERVICE_TOKEN=
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
AI_REQUEST_TIMEOUT_SECONDS=30
```

Set `AI_SERVICE_TOKEN` to a private token shared **only** with the backend's `AI_SERVICE_TOKEN`, and configure the backend's `AI_SERVICE_URL` to reach this service. `GEMINI_API_KEY` must be kept on the server. `GEMINI_MODEL` defaults to `gemini-2.5-flash` and the provider timeout defaults to 30 seconds. The application reads process environment variables directly; copying the example to `.env` alone does not load it. Do not commit credentials or paste them into request examples.

With the variables present in your shell, start the local API:

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

`GET http://localhost:8000/health` responds without an analysis token. `POST /analyze` is an internal endpoint requiring `Authorization: Bearer <AI_SERVICE_TOKEN>` and uncompressed `application/json` matching `AnalysisRequest` in `schemas.py`. FastAPI's Swagger UI, ReDoc, and OpenAPI endpoints are disabled; use the schemas and the backend's reference-only `/api/docs` for integration context.

## Technical documentation

- `main.py` defines `/health`, token authentication, request-size and timeout limits, Gemini calls, provider error handling, and the `/analyze` route. An invalid or missing service configuration returns 503; missing/invalid credentials return 401.
- `schemas.py` defines the evidence input and structured output contracts, source provenance, version identifiers, and validation rules. Do not infer verified facts from missing evidence.
- `/analyze` accepts at most 128 KiB of uncompressed JSON. Requests are subject to per-process concurrency and rate limits; successful responses are not cached.
- `pyproject.toml` and `uv.lock` pin application dependencies; `Dockerfile` packages the service for port 8000. Verify container bind settings in your deployment environment before using the Dockerfile's default command.
- Local quality checks: `uv run ruff check .` and `uv run mypy`. Do not run `diagnostic_probe.py` against real services or private environment files as a routine setup step.

## Related repositories

[Organization](https://github.com/Blazemap) · [Frontend](https://github.com/Blazemap/Blazemap-FE) · [Backend](https://github.com/Blazemap/Blazemap-BE)
