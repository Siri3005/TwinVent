# Owner 1 model to Owner 2 app integration

The first parallel-workflow contract proposed an `artifacts/model/predictor.py` module. Owner 1's completed work provides a `PressurePredictor` class in `src/model_training/inference_wrapper.py` and a trained artifact at `artifacts/model/final_model.pkl`. The app now bridges to that real interface through `src/predictor_adapter/benchmark.py`; no extra model module is needed.

## Data passed to the model

For one selected breath, the app sends a list of ordered dictionaries containing only:

| Field | Meaning |
| --- | --- |
| `id` | Original CSV sample ID; used to align the returned curve. |
| `time_step` | Time within the breath, in seconds. |
| `u_in` | Inspiratory input signal. |
| `u_out` | Valve state, 0 or 1. |
| `R` | Artificial test-lung resistance. Required by this model. |
| `C` | Artificial test-lung compliance. Required by this model. |

The measured `pressure` and dataset `breath_id` are not sent to the predictor. This avoids target leakage and keeps the call scoped to one breath. The dataset reader sorts each breath by `time_step`, checks the 80-row shape and constant R/C values, and rejects missing mechanics rather than filling them in.

## Owner 1 API

The adapter loads `artifacts/model/final_model.pkl` through Owner 1's wrapper:

```python
from model_training.inference_wrapper import PressurePredictor

predictor = PressurePredictor(model_path="artifacts/model/final_model.pkl")
predictor.load_model()
prediction = predictor.predict_breath(breath_dataframe, return_uncertainty=False)
```

The wrapper returns a DataFrame with `id`, `pressure`, `status`, `reason`, and `model_version`. The adapter converts this into the app response below. The model is loaded lazily and reused for subsequent breaths.

## App response

Successful response:

```json
{
  "row_ids": [101, 102],
  "pressure": [5.8, 6.2],
  "uncertainty": [null, null],
  "status": "ok",
  "reason": null,
  "model_version": "1.0.0"
}
```

The arrays contain one value per input row. IDs must match exactly and in order, and pressure values must be finite. The adapter does not expose Owner 1's current uncertainty value because the wrapper describes it as a placeholder based on prediction magnitude, not a calibrated uncertainty estimate.

When Owner 1's wrapper returns `abstain`, the adapter returns the matching row IDs with empty pressure and uncertainty lists. The UI clears any previous curve and displays the model's reason. Missing model files or runtime dependencies produce an `unavailable` response. A model execution or output-contract error produces an error response; neither case shows a partial curve.

## Runtime requirements

The app and mock predictor use the Python standard library. To run the Owner 1 model, install its local dependencies from the repository root before going offline:

```powershell
python -m pip install -r requirements.txt
```

No network call is made by the app or predictor during inference. The browser app binds only to `127.0.0.1`.

## Scope and interpretation

The pressure unit follows the benchmark specification's assumed cmH₂O scale. Owner 1's model was trained on artificial test-lung data and predicts the pressure represented by the provided input samples. It does not represent an identified patient's lung, estimate ARDS status, predict outcomes, or simulate a validated clinical setting intervention. Do not describe this integration as a patient-specific digital twin or use it to guide ventilator settings.
