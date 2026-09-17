import asyncio
import os
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from fastapi import HTTPException
from google.genai import types

from main import ProviderConfig, call_provider, provider_config
from schemas import Analysis, AnalysisRequest, Reason


class FakeModels:
    def __init__(self, result: Analysis) -> None:
        self.result = result
        self.call: dict[str, object] = {}

    async def generate_content(self, **kwargs: object) -> object:
        self.call = kwargs
        return SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
            function_calls=None,
            model_version="gemini-check",
            text=self.result.model_dump_json(),
        )


async def check() -> None:
    context = AnalysisRequest(
        caseId="case-1",
        contextRevision=1,
        verificationStatus="UNVERIFIED",
        observations=[
            {
                "id": "observation-1",
                "kind": "community-report",
                "observedAt": datetime.now(UTC),
                "description": "Smoke reported",
            }
        ],
        weather=None,
        spatialContext=[],
        operationalContext=[],
    )
    analysis = Analysis(
        evidenceLevel="LOW",
        impactLevel="INSUFFICIENT_DATA",
        suggestedPriority="LOW",
        reasons=[Reason(text="One unverified report was supplied.", sourceIds=["observation-1"])],
        missingInformation=["Independent evidence is missing."],
        monitoringAreas=[],
        suggestedChecks=["Consider obtaining independent observations."],
        limitations=["The report is not independently verified."],
    )
    models = FakeModels(analysis)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    config = ProviderConfig("placeholder", "gemini-2.5-flash", 5)
    response = await call_provider(context, config, client=cast(Any, client))
    generation = cast(types.GenerateContentConfig, models.call["config"])
    assert response.caseId == context.caseId
    assert response.contextRevision == context.contextRevision
    assert response.model == "gemini-check"
    assert models.call["model"] == config.model
    assert isinstance(generation, types.GenerateContentConfig)
    assert generation.response_mime_type == "application/json"
    assert generation.response_schema is Analysis
    assert generation.temperature == 0
    assert config.api_key not in repr(config)
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": "gemini-2.5-flash"}):
        try:
            provider_config()
        except HTTPException as error:
            assert error.status_code == 503
        else:
            raise AssertionError("Missing Gemini credentials must fail closed")


asyncio.run(check())
print("Gemini provider contract checks passed without network access.")
