# TwinVent offline benchmark replay

This repository contains the supplied artificial test-lung CSV data, the TwinVent project plan, two parallel owner workflows, and Owner 2's first offline replay application.

## Run the replay app

1. Install Python 3.10 or newer. The replay app uses only the Python standard library and a modern browser; it does not need packages, cloud services, or an internet connection.
2. From this repository folder, run `python run_app.py` in PowerShell or a terminal.
3. Open `http://127.0.0.1:8765` on the same computer.
4. Select `train.csv` or `test.csv`, enter a breath ID, load the breath, and choose **Generate prediction**.
5. Select **Mock demonstration** for an illustrative, hand-written curve. **Owner 1 benchmark model** stays unavailable until a compatible model is placed at `artifacts/model/predictor.py`.
6. Stop the server with Ctrl+C in the terminal.

The first dataset selection scans the chosen CSV to build a local index. Later breath lookups use the generated index under `.twinvent/cache/`. The source files are read in place and are not copied. The app log stores only source, breath ID, status, model version, and time; it does not store waveform values. Use **Clear local index and app log** to remove generated index files and the local event log.

## Scope and limits

The data describes an artificial test lung and is useful for a pressure-curve software demonstration. The mock predictor is not trained. The dataset has no patient histories, ARDS outcomes, or ventilator-setting interventions. This app is not patient-specific clinical decision support, has no live ventilator interface, and cannot control a ventilator.

See `docs/app/README.md` for the Owner 2 workflow and `docs/app/INTERFACE_CONTRACT.md` for the model handoff format.
