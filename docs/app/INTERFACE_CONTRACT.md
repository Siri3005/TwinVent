# TwinVent predictor interface contract

This contract lets Owner 2 build the replay app while Owner 1 develops the benchmark model. The app uses the mock implementation until the real model artifact is available.

## Input

The app calls `predict_breath(rows)` once for a selected breath. `rows` is an ordered list of 80 dictionaries. Every dictionary contains only these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | integer | Original CSV row ID; output must preserve this exact order. |
| `time_step` | finite number | Time within the breath. |
| `u_in` | finite number | Dataset input signal. |
| `u_out` | integer, 0 or 1 | Dataset output-valve state. |
| `R` | finite number or `None` | Artificial test-lung resistance if supplied. |
| `C` | finite number or `None` | Artificial test-lung compliance if supplied. |

The input intentionally excludes measured `pressure`; this prevents the target from leaking into prediction features. R and C are optional at the app boundary for future adapters, but Owner 1's first benchmark may require them. The app must never invent or impute them silently.

## Model artifact

Place the Python module at `artifacts/model/predictor.py`. It must export a `predict_breath(rows)` function and may export `MODEL_VERSION = "model-name-version"`. It must run locally without internet access and return within a short interactive response time.

## Successful output

Return a dictionary (or `PredictionResult`) with this structure:

```python
{
    "row_ids": [101, 102],
    "pressure": [5.8, 6.2],
    "uncertainty": [None, None],
    "status": "ok",
    "reason": None,
    "model_version": "benchmark-v1",
}
```

The example is abbreviated; a real result must contain one ID, pressure, and uncertainty entry for every input row. `row_ids` must match the input `id` values exactly and in order. Pressure values must be finite numbers. `uncertainty` is optional; if omitted, the app fills it with `None` values.

## Non-success status

Use `status` equal to `abstain` when the model deliberately declines to predict. `reason` should state why. The app also understands `unavailable` and `error` for adapter/startup failures. For a non-success status, return empty pressure and uncertainty lists; the app must not show a partial or stale curve.

## Unit and scope

Pressure is plotted as cmH2O for this competition dataset. Preserve the data's numerical scale and document any preprocessing in the model package. This is an artificial test-lung benchmark interface, not a clinical API or ventilator-control interface.
