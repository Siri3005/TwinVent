"""Adapter for the future Owner 1 benchmark model artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from .contracts import PredictionError, PredictionResult, PredictorUnavailable


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "model" / "predictor.py"


class BenchmarkModelAdapter:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.is_file():
            raise PredictorUnavailable(
                "The Owner 1 benchmark model is not installed yet. Use the mock for a demo, "
                "or place artifacts/model/predictor.py in the agreed format."
            )
        spec = importlib.util.spec_from_file_location("twinvent_benchmark_predictor", model_path)
        if spec is None or spec.loader is None:
            raise PredictorUnavailable("The benchmark model file could not be loaded.")
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(spec.name, None)
            raise PredictorUnavailable(f"The benchmark model could not be initialized: {exc}") from exc
        predict = getattr(module, "predict_breath", None)
        if not callable(predict):
            raise PredictorUnavailable("The model file must provide predict_breath(rows).")
        self._predict = predict
        self.model_version = str(getattr(module, "MODEL_VERSION", "benchmark-unversioned"))

    def predict_breath(self, rows: list[dict[str, Any]]) -> PredictionResult:
        try:
            raw = self._predict(rows)
        except Exception as exc:
            raise PredictionError(f"Benchmark model failed: {exc}") from exc
        if isinstance(raw, PredictionResult):
            return raw
        if not isinstance(raw, dict):
            raise PredictionError("Benchmark model must return a prediction dictionary.")
        try:
            ids = [int(value) for value in raw["row_ids"]]
            pressure = [None if value is None else float(value) for value in raw["pressure"]]
            uncertainty = raw.get("uncertainty")
            if uncertainty is None:
                uncertainty = [None] * len(ids)
            else:
                uncertainty = [None if value is None else float(value) for value in uncertainty]
            return PredictionResult(
                row_ids=ids,
                pressure=pressure,
                uncertainty=uncertainty,
                status=str(raw.get("status", "ok")),
                reason=raw.get("reason"),
                model_version=str(raw.get("model_version", self.model_version)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionError("Benchmark model returned an invalid result structure.") from exc
