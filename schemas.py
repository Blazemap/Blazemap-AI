import json
import math
import re
import unicodedata
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

MAX_SOURCES = 50
MAX_TEXT = 2000
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^\S+$")]
Text = Annotated[str, Field(min_length=1, max_length=MAX_TEXT, pattern=r"\S")]
Level = Literal["LOW", "MODERATE", "HIGH", "INSUFFICIENT_DATA"]
SourceIds = Annotated[list[Identifier], Field(min_length=1, max_length=MAX_SOURCES)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Observation(StrictModel):
    id: Identifier
    kind: Annotated[str, Field(min_length=1, max_length=80, pattern=r"\S")]
    observedAt: AwareDatetime
    description: Annotated[str, Field(max_length=MAX_TEXT)] = ""
    latitude: Annotated[float, Field(ge=-90, le=90)] | None = None
    longitude: Annotated[float, Field(ge=-180, le=180)] | None = None


class SpatialSource(StrictModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    id: Identifier
    name: Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")] | None = None
    relationBasis: (
        Literal["ADMINISTRATIVE_REGION_ONLY", "VERIFIED_DISTANCE", "VERIFIED_DOWNWIND"] | None
    ) = None
    computedBy: Literal["BLAZEMAP"] | None = None
    distanceMeters: Annotated[float, Field(ge=0)] | None = None
    sourceId: Identifier | None = None
    sourceDate: AwareDatetime | None = None
    downwind: bool | None = None
    forecastId: Identifier | None = None

    @model_validator(mode="after")
    def exposure_facts(self) -> Self:
        if self.relationBasis in ("VERIFIED_DISTANCE", "VERIFIED_DOWNWIND"):
            if (
                self.computedBy != "BLAZEMAP"
                or self.distanceMeters is None
                or self.sourceId is None
                or self.sourceDate is None
            ):
                raise ValueError("Verified exposure requires computation and source provenance")
            if self.relationBasis == "VERIFIED_DOWNWIND":
                if self.downwind is not True or self.forecastId is None:
                    raise ValueError("Verified downwind requires a positive relation and forecast")
            elif self.downwind is not None or self.forecastId is not None:
                raise ValueError("Distance evidence cannot assert a downwind relationship")
        return self


class AnalysisRequest(StrictModel):
    caseId: Identifier
    contextRevision: Annotated[int, Field(ge=0)]
    verificationStatus: Literal["UNVERIFIED", "CONFIRMED_FIRE", "NOT_FIRE"]
    observations: Annotated[list[Observation], Field(min_length=1, max_length=MAX_SOURCES)]
    weather: dict[str, JsonValue] | None
    spatialContext: Annotated[list[SpatialSource], Field(max_length=MAX_SOURCES)]
    operationalContext: Annotated[list[dict[str, JsonValue]], Field(max_length=MAX_SOURCES)]

    @model_validator(mode="after")
    def source_identity(self) -> Self:
        sources = list(self.operationalContext)
        if self.weather is not None:
            sources.append(self.weather)
        observation_ids = {observation.id for observation in self.observations}
        ids = [observation.id for observation in self.observations]
        ids.extend(source.id for source in self.spatialContext)
        for source in sources:
            identity = source.get("id")
            if not isinstance(identity, str) or not re.fullmatch(r"\S{1,128}", identity):
                raise ValueError("Each context source requires an id")
            ids.append(identity)
        if len(ids) > MAX_SOURCES or len(ids) != len(set(ids)):
            raise ValueError("Source IDs must be unique with at most 50 sources in total")
        for spatial in self.spatialContext:
            if spatial.relationBasis in ("VERIFIED_DISTANCE", "VERIFIED_DOWNWIND"):
                if spatial.sourceId not in observation_ids:
                    raise ValueError("Exposure sourceId must reference a supplied observation")
                if spatial.relationBasis == "VERIFIED_DOWNWIND" and (
                    self.weather is None or spatial.forecastId != self.weather.get("id")
                ):
                    raise ValueError("Downwind forecastId must reference supplied weather")
        return self

    def source_ids(self) -> set[str]:
        sources = list(self.operationalContext)
        if self.weather is not None:
            sources.append(self.weather)
        return (
            {item.id for item in self.observations}
            | {source.id for source in self.spatialContext}
            | {str(source["id"]) for source in sources}
        )


class Reason(StrictModel):
    text: Text
    sourceIds: SourceIds


class MonitoringArea(StrictModel):
    name: Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]
    reason: Text
    sourceIds: SourceIds


class Analysis(StrictModel):
    evidenceLevel: Level
    impactLevel: Level
    suggestedPriority: Literal["HIGH", "MEDIUM", "LOW", "UNASSESSED"]
    reasons: Annotated[list[Reason], Field(min_length=1, max_length=20)]
    missingInformation: Annotated[list[Text], Field(max_length=20)]
    monitoringAreas: Annotated[list[MonitoringArea], Field(max_length=20)]
    suggestedChecks: Annotated[list[Text], Field(max_length=20)]
    limitations: Annotated[list[Text], Field(min_length=1, max_length=20)]


class AnalysisResponse(Analysis):
    caseId: Identifier
    contextRevision: Annotated[int, Field(ge=0)]
    model: Annotated[str, Field(min_length=1, max_length=200, pattern=r"^\S+$")]
    generatedAt: AwareDatetime


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def load_json(value: str | bytes) -> Any:
    return json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant)


def bound_context(value: Any, depth: int = 0) -> None:
    if depth > 8:
        raise ValueError("Context nesting exceeds limit")
    if isinstance(value, dict):
        if len(value) > 50 or any(len(key) > 64 for key in value):
            raise ValueError("Context object exceeds limit")
        for child in value.values():
            bound_context(child, depth + 1)
    elif isinstance(value, list):
        if len(value) > MAX_SOURCES:
            raise ValueError("Context list exceeds limit")
        for child in value:
            bound_context(child, depth + 1)
    elif isinstance(value, str) and len(value) > MAX_TEXT:
        raise ValueError("Context text exceeds limit")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite context number")


DENIED = re.compile(
    r"\b(?:evacuat\w*|dispatch\w*|deploy\w*|warn(?:ing|ings)?|perimeters?|"
    r"firebreaks?|suppress(?:ion)?|publish\w*)\b"
    r"|\b(?:issue|send|order)\b[^.!?\n]{0,60}\b(?:alerts?|teams?|responders?)\b"
    r"|\b(?:fire|incident)\s+(?:(?:is|was|has been)\s+)?(?:confirmed|verified)\b"
    r"|\bconfirmed\s+(?:fire|incident)\b|\bno\s+fire\b"
    r"|\b(?:all\s+clear|no\s+risk|safe\s+route|(?:area|village|road)\s+is\s+safe)\b"
    r"|\bsmoke\b[^.!?\n]{0,160}\b(?:arriv\w*|reach\w*|eta|\d+\s*(?:minutes?|hours?))\b"
    r"|\d+(?:\.\d+)?\s*%|<[^>]+>",
    re.IGNORECASE,
)


def validate_analysis(result: Analysis, context: AnalysisRequest) -> None:
    allowed = context.source_ids()
    citations: set[str] = set()
    references = [item.sourceIds for item in result.reasons]
    references.extend(item.sourceIds for item in result.monitoringAreas)
    for source_ids in references:
        ids = set(source_ids)
        if len(ids) != len(source_ids) or not ids <= allowed:
            raise ValueError("Invalid source references")
    for reason in result.reasons:
        citations.update(reason.sourceIds)
    if result.evidenceLevel != "INSUFFICIENT_DATA" and not citations.intersection(
        observation.id for observation in context.observations
    ):
        raise ValueError("Evidence assessment requires observation references")
    exposure_sources = {
        source.id: source
        for source in context.spatialContext
        if source.relationBasis in ("VERIFIED_DISTANCE", "VERIFIED_DOWNWIND")
    }
    if result.impactLevel != "INSUFFICIENT_DATA" and not citations.intersection(exposure_sources):
        raise ValueError("Impact assessment requires cited deterministic exposure evidence")
    if (
        result.evidenceLevel == result.impactLevel == "INSUFFICIENT_DATA"
        and result.suggestedPriority != "UNASSESSED"
    ):
        raise ValueError("Insufficient data cannot support assessed priority")
    for area in result.monitoringAreas:
        if not any(
            source.id in area.sourceIds and source.name == area.name
            for source in exposure_sources.values()
        ):
            raise ValueError("Monitoring requires named cited deterministic exposure evidence")
    texts = [reason.text for reason in result.reasons]
    texts.extend(text for area in result.monitoringAreas for text in (area.name, area.reason))
    texts.extend(result.missingInformation + result.suggestedChecks + result.limitations)
    for text in texts:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFKC", text)
            if unicodedata.category(character) != "Cf"
        )
        # ponytail: lexical checks only; semantic grounding still requires human review.
        if DENIED.search(normalized):
            raise ValueError("Unsupported output")
