# Owner 2 offline replay app

## What is implemented

- A local browser app served only on `127.0.0.1`.
- Read-only breath replay from the repository's `train.csv` and `test.csv` files.
- A disk-cached index of breath IDs and byte offsets, so each selection reads only one breath rather than loading a multi-gigabyte table into memory.
- Checks for required columns, malformed numeric values, duplicated row IDs, wrong breath membership, sample count, repeated timestamps, invalid `u_out`, inconsistent R/C values, and incomplete measured pressure.
- Measured and predicted pressure charting, the `u_in` input curve, and `u_out` phase shading. The charts are drawn with the browser canvas and use no remote library.
- A deterministic mock predictor and an adapter to Owner 1's completed `PressurePredictor` plus `final_model.pkl` artifact.
- Output checks for row ID alignment, curve length, finite values, model status, and model version. An unavailable or failed model does not leave an old predicted curve visible.
- Local-only event logging and a control to clear the generated index and app log without changing the CSV data.
- No live ventilator connection, control command, internet dependency, or cloud transmission.

## Start

From the repository root:

```powershell
python run_app.py
```

Then open `http://127.0.0.1:8765`. Use `python run_app.py --open-browser` to ask Python to open the page automatically. Use `python run_app.py --port 8766` if port 8765 is already occupied. Press Ctrl+C to stop the app.

The initial scan creates a local index and can take time because the data files are large. Later selections use the cache. If the cache is cleared, the next selection scans the file again. The app reads source CSVs in place; it does not create duplicate copies.

The mock mode needs no installed packages. To use the trained Owner 1 model, install the repository requirements locally before disconnecting from the internet:

```powershell
python -m pip install -r requirements.txt
```

## Demo flow

1. Select `train.csv`, load breath `1`, and use **Mock demonstration**. The dark line is measured pressure; the dashed orange line is the mock output.
2. Select **Owner 1 benchmark model** to use `artifacts/model/final_model.pkl` through the completed wrapper in `src/model_training/inference_wrapper.py`. On a training breath, the app shows the measured and predicted curves and calculates a per-breath mean absolute difference. This is not the validation score.
3. Select `test.csv`, load breath `0`, and generate a benchmark prediction. There is no measured-pressure line or error metric because test data has no target labels.
4. The adapter passes only `id`, `time_step`, `u_in`, `u_out`, `R`, and `C`. It keeps sample IDs aligned and does not show a partial curve if the model abstains or fails.
5. Enter a missing ID or select a malformed file to see a visible error. The source data is never edited.

## Local storage

- `.twinvent/cache/` contains only index metadata: file size, modification time, breath IDs, row counts, and byte offsets.
- `.twinvent/logs/session.jsonl` records the time and source, breath ID, result status, and model version for a small local audit trail. It never records pressure, flow input values, or the complete breath.
- The clear button deletes generated JSON index files and the local event log after browser confirmation. It never deletes or rewrites CSV files.

## Owner 1 model integration

Owner 1 has completed the model package and interface specification. Owner 2's `BenchmarkModelAdapter` bridges the app's row-list interface to `PressurePredictor.predict_breath(DataFrame)` and loads `artifacts/model/final_model.pkl` once for reuse. See `INTERFACE_CONTRACT.md` for the input mapping, output shape, validation behavior, and runtime dependencies.

## Important scope note

The hand-written mock curve is only for app wiring and demonstration. It is not trained on the dataset and is not an estimate of a patient's lung state. Owner 1's trained model is also limited to the artificial test-lung benchmark; neither mode identifies a patient, simulates a validated setting intervention, or validates clinical benefit or setting-change safety.
