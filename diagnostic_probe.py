import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
from google.genai import errors

from main import call_provider, provider_client, provider_config
from schemas import AnalysisRequest


def load_env(path):
    return {
        k.strip(): v.strip().strip('"').strip("'")
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
        for k, v in [line.split("=", 1)]
    }


async def run():
    ai = load_env(Path(".env"))
    be = load_env(Path("../Blazemap-BE/.env"))
    os.environ.update(ai)
    print(
        json.dumps(
            {
                "tokenMatch": bool(ai.get("AI_SERVICE_TOKEN"))
                and ai.get("AI_SERVICE_TOKEN") == be.get("AI_SERVICE_TOKEN"),
                "providerKeyPresent": bool(ai.get("GEMINI_API_KEY")),
                "serviceHost": __import__("urllib.parse", fromlist=["urlsplit"])
                .urlsplit(be["AI_SERVICE_URL"])
                .hostname,
                "model": ai.get("GEMINI_MODEL", "gemini-2.5-flash"),
            }
        )
    )
    context = AnalysisRequest(
        caseId="synthetic-diagnostic-not-an-incident",
        contextRevision=1,
        verificationStatus="UNVERIFIED",
        coordinatePrecision={
            "decimalPlaces": 2,
            "method": "DECIMAL_ROUNDING",
            "exactCoordinatesShared": False,
        },
        observations=[
            {
                "id": "synthetic-observation",
                "kind": "community-report",
                "observedAt": datetime.now(UTC),
                "description": (
                    "Synthetic diagnostic only, not a real incident. Unverified smoke indication."
                ),
            }
        ],
        weather=None,
        spatialContext=[],
        operationalContext=[],
    )
    async with httpx.AsyncClient(timeout=70, follow_redirects=False, trust_env=False) as client:
        try:
            response = await client.post(
                be["AI_SERVICE_URL"].rstrip("/") + "/analyze",
                headers={"Authorization": "Bearer " + be["AI_SERVICE_TOKEN"]},
                json=json.loads(context.model_dump_json()),
            )
            print(
                json.dumps(
                    {
                        "stage": "configured-service",
                        "status": response.status_code,
                        "schemaVersion": response.headers.get("X-AI-Schema-Version"),
                        "success": response.status_code == 200,
                    }
                )
            )
        except Exception:
            print(json.dumps({"stage": "configured-service", "code": "CONNECTION_FAILED"}))
    config = provider_config()
    provider = provider_client(config)
    try:
        await provider.aio.models.get(model=config.model)
        print(json.dumps({"stage": "provider-model", "code": "MODEL_ACCESS_OK"}))
    except errors.APIError as error:
        print(json.dumps({"stage": "provider-model", "status": error.code}))
    except Exception:
        print(json.dumps({"stage": "provider-model", "code": "CONNECTION_FAILED"}))
    original = provider.aio.models.generate_content

    async def diagnostic_generate(**kwargs):
        try:
            return await original(**kwargs)
        except errors.APIError as error:
            message = str(error).lower()
            print(
                json.dumps(
                    {
                        "stage": "generation-http",
                        "status": error.code,
                        "schemaRelated": "schema" in message,
                        "unsupportedFields": [
                            key
                            for key in [
                                "additionalproperties",
                                "pattern",
                                "minlength",
                                "maxlength",
                                "$ref",
                            ]
                            if key in message
                        ],
                    }
                )
            )
            raise

    provider.aio.models.generate_content = diagnostic_generate
    try:
        result = await call_provider(context, config, client=provider)
        print(
            json.dumps({"stage": "provider-analysis", "code": "SUCCEEDED", "model": result.model})
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "stage": "provider-analysis",
                    "status": getattr(error, "status_code", None),
                    "detail": getattr(error, "detail", None),
                    "code": type(error).__name__,
                    "frames": [
                        frame.name
                        for frame in __import__("traceback").extract_tb(error.__traceback__)
                    ],
                }
            )
        )


asyncio.run(run())
