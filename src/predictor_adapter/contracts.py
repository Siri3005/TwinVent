"""Shared prediction interface for Owner 1's model and Owner 2's app."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol


@dataclass(frozen=True)
class PredictionResult:
    row_ids: list[int]
    pressure: list[float | None]
    uncertainty: list[float | None]
    status: str
    reason: str | None
    model_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_ids": self.row_ids,
            "pressure": self.pressure,
            "uncertainty": self.uncertainty,
            "status": self.status,
            "reason": self.reason,
            "model_version": self.model_version,
        }


class Predictor(Protocol):
    model_version: str

    def predict_breath(self, rows: list[dict[str, Any]]) -> PredictionResult:
        """Return one result for each input row, keeping input row ID alignment."""


class PredictorUnavailable(RuntimeError):
    pass


class PredictionError(RuntimeError):
    pass


def validate_prediction(result: PredictionResult, rows: list[dict[str, Any]]) -> None:
    ids = [int(row["id"]) for row in rows]
    if result.status not in {"ok", "abstain", "unavailable", "error"}:
        raise PredictionError("Predictor returned an unknown status.")
    if result.row_ids != ids:
        raise PredictionError("Predictor output row IDs do not match the selected breath.")
    if result.status == "ok":
        if len(result.pressure) != len(rows) or len(result.uncertainty) != len(rows):
            raise PredictionError("Predictor returned the wrong number of values.")
        for value in result.pressure:
            if value is None or not math.isfinite(value):
                raise PredictionError("Predictor returned a missing or non-finite pressure value.")
        for value in result.uncertainty:
            if value is not None and not math.isfinite(value):
                raise PredictionError("Predictor returned a non-finite uncertainty value.")
    elif result.pressure and len(result.pressure) != len(rows):
        raise PredictionError("Predictor returned a partial curve for a non-ok status.")
