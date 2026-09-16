# Blazemap AI

An internal decision-support service for Blazemap's forest and land fire awareness system in Kalimantan, Indonesia.

## Purpose

Returns structured, source-referenced analysis for human review, including uncertainty, missing evidence, and limitations.

## Tech Stack

| Area | Technology |
| --- | --- |
| Language | Python 3.12+ |
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| Provider Client | HTTPX, OpenAI-compatible API |
| Dependencies | uv |

## Boundaries

AI does not confirm fires, publish official warnings, dispatch teams, or order evacuations. The service requires an authenticated backend request and configured model provider.

## Related Repositories

[Organization](https://github.com/Blazemap) · [Frontend](https://github.com/Blazemap/Blazemap-FE) · [Backend](https://github.com/Blazemap/Blazemap-BE)
