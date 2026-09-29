"""Deterministic illustrative output. This is not a trained or clinical model."""

from __future__ import annotations

import math
from typing import Any

from .contracts import PredictionResult


class MockPredictor:
    model_version = "mock-demo-v1"

    def predict_breath(self, rows: list[dict[str, Any]]) -> PredictionResult:
        if not rows:
            return PredictionResult([], [], [], "abstain", "No rows were provided.", self.model_version)
        pressure: list[float] = []
        ids: list[int] = []
        max_time = max(float(row["time_step"]) for row in rows) or 1.0
        for row in rows:
            u_in = float(row["u_in"])
            r_value = row.get("R")
            c_value = row.get("C")
            r_term = 0.018 * (float(r_value) - 20.0) if r_value is not None else 0.0
            c_term = -0.006 * (float(c_value) - 20.0) if c_value is not None else 0.0
            if int(row["u_out"]) == 1:
                value = 5.0 + 0.035 * u_in + 0.01 * (float(r_value) if r_value is not None else 20.0)
            else:
                value = 5.0 + 0.40 * u_in + r_term + c_term + 0.15 * math.sin(2.0 * math.pi * float(row["time_step"]) / max_time)
            ids.append(int(row["id"]))
            pressure.append(round(value, 5))
        return PredictionResult(
            row_ids=ids,
            pressure=pressure,
            uncertainty=[None] * len(rows),
            status="ok",
            reason="Illustrative demo curve only; this mock is not trained or clinically calibrated.",
            model_version=self.model_version,
        )
