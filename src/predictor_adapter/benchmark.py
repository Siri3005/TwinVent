"""Bridge Owner 1's PressurePredictor to Owner 2's offline app contract."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from .contracts import PredictionError, PredictionResult, PredictorUnavailable


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "model" / "lstm_model.h5"
MODEL_VERSION = "3.0.0-lstm"
_LOAD_LOCK = threading.Lock()


class BenchmarkModelAdapter:
    """Load the trained artifact once and adapt its DataFrame API for the app.

    Owner 1 supplies a PressurePredictor class instead of the temporary
    ``predictor.py`` interface originally sketched for parallel development.
    This adapter keeps that package detail inside the backend and exposes the
    stable row-dictionary interface used by the app.
    """

    _predictor: Any = None

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.is_file():
            raise PredictorUnavailable(
                "The Owner 1 trained model is missing. Expected "
                "artifacts/model/final_model.pkl."
            )
        cls = type(self)
        if cls._predictor is None:
            with _LOAD_LOCK:
                if cls._predictor is None:
                    try:
                        from model_training.lstm_inference_wrapper import LSTMPressurePredictor

                        predictor = LSTMPressurePredictor(model_path=str(model_path))
                        predictor.load_model()
                        cls._predictor = predictor
                    except Exception as exc:
                        raise PredictorUnavailable(
                            "The Owner 1 model could not be loaded. Install its local "
                            "runtime dependencies from requirements.txt and check the "
                            f"model artifact. Details: {exc}"
                        ) from exc
        self._predict = cls._predictor
        self.model_version = str(self._predict.VERSION or MODEL_VERSION)

    def predict_breath(self, rows: list[dict[str, Any]]) -> PredictionResult:
        if not rows:
            return PredictionResult([], [], [], "abstain", "No rows were provided.", self.model_version)
        try:
            import pandas as pd

            # Do not include pressure or breath_id: the model only needs its
            # documented features, and the measured target must not leak in.
            frame = pd.DataFrame(rows, columns=["id", "time_step", "u_in", "u_out", "R", "C"])
            output = self._predict.predict_breath(frame, return_uncertainty=False)
            ids = [int(value) for value in output["id"].tolist()]
            statuses = set(str(value) for value in output["status"].tolist())
            reason_values = [str(value) for value in output["reason"].tolist() if value and str(value) != "nan"]
            reason = "; ".join(dict.fromkeys(reason_values)) or None

            if statuses == {"abstain"}:
                return PredictionResult(ids, [], [], "abstain", reason or "The benchmark model abstained.", self.model_version)
            if statuses != {"ok"}:
                raise PredictionError("Owner 1 model returned mixed or unknown row statuses.")

            pressure = [float(value) for value in output["pressure"].tolist()]
            return PredictionResult(
                row_ids=ids,
                pressure=pressure,
                # Owner 1's optional uncertainty is currently a placeholder,
                # so do not present it as a calibrated confidence estimate.
                uncertainty=[None] * len(ids),
                status="ok",
                reason=reason,
                model_version=self.model_version,
            )
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(f"Owner 1 benchmark model failed: {exc}") from exc
