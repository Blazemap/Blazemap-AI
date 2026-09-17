import json
import unittest

from pydantic import ValidationError

from schemas import DENIED, AnalysisRequest, WindContext


class WindChecks(unittest.TestCase):
    def test_context(self) -> None:
        value = {
            "status": "READY",
            "evaluatedAt": "2026-09-17T10:00:00Z",
            "usableUntil": "2026-09-17T12:00:00Z",
            "forecast": {
                "id": "forecast",
                "provider": "BMKG",
                "regionId": "region",
                "regionName": "Village",
                "issuedAt": "2026-09-17T06:00:00Z",
                "validAt": "2026-09-17T09:00:00Z",
                "fetchedAt": "2026-09-17T08:00:00Z",
            },
            "windSpeedKmh": 12,
            "windFromDegrees": 0,
            "windToDegrees": 180,
            "summary": "Wind from N toward S.",
            "disclaimer": "Downwind attention, not predicted perimeter",
            "spatialExtent": None,
            "settlementExposure": "UNAVAILABLE",
            "ruleVersion": "wind-context-1",
        }
        self.assertEqual(WindContext.model_validate_json(json.dumps(value)).windToDegrees, 180)
        for patch in (
            {"windToDegrees": 90},
            {"windSpeedKmh": 0},
            {"forecast": None},
            {"usableUntil": None},
            {"spatialExtent": {"radius": 100}},
            {"usableUntil": "2026-09-18T12:00:00Z"},
        ):
            with self.assertRaises(ValidationError):
                WindContext.model_validate_json(json.dumps(value | patch))
        calm = value | {
            "status": "CALM",
            "windSpeedKmh": 0,
            "windFromDegrees": None,
            "windToDegrees": None,
        }
        self.assertEqual(WindContext.model_validate_json(json.dumps(calm)).status, "CALM")
        request = {
            "caseId": "case",
            "contextRevision": 1,
            "verificationStatus": "CONFIRMED_FIRE",
            "observations": [
                {"id": "field", "kind": "FIELD_UPDATE", "observedAt": "2026-09-17T09:00:00Z"}
            ],
            "weather": {
                "id": "forecast",
                "windSpeed": 12,
                "windSpeedUnit": "km/h",
                "windFromDegrees": 0,
                "windToDegrees": 180,
            },
            "windContext": value,
            "spatialContext": [],
            "operationalContext": [],
        }
        self.assertIsNotNone(AnalysisRequest.model_validate_json(json.dumps(request)).windContext)
        with self.assertRaises(ValidationError):
            AnalysisRequest.model_validate_json(json.dumps(request | {"weather": None}))
        for text in (
            "Fire will spread 2 km",
            "Wildfire reaches the village in an hour",
            "Spread speed is 10 km/h",
        ):
            self.assertIsNotNone(DENIED.search(text))


if __name__ == "__main__":
    unittest.main()
