"use strict";

const $ = (id) => document.getElementById(id);
const state = { metadata: null, breath: null, prediction: null, scenario: null, twinCondition: null, busy: false, replayTimer: null };

function setStatus(message, kind = "neutral") {
  const badge = $("statusBadge");
  badge.className = `status-badge status-${kind}`;
  $("statusText").textContent = message;
}

function clearPrediction() {
  state.prediction = null;
  state.scenario = null;
  $("resultStatus").textContent = "Not run";
  $("resultMessage").textContent = "Choose a breath and generate a prediction.";
  $("peakMetric").textContent = "—";
  $("maeMetric").textContent = "—";
  $("uncertaintyMetric").textContent = "—";
  $("scenarioButton").disabled = true;
  $("scenarioSummary").textContent = "Load and predict a train.csv breath first.";
  $("scenarioState").textContent = "Simulation ready";
  $("effectStatus").textContent = "Run a scenario to compare";
  $("currentPeakMetric").textContent = "—";
  $("proposedPeakMetric").textContent = "—";
  $("peakChangeMetric").textContent = "—";
  $("scenarioExplanation").textContent = "Run a scenario to compare current and proposed pressure predictions.";
  $("predictionDetails").hidden = true;
  $("effectStatus").textContent = "Run a scenario to compare";
  drawScenario();
  drawCharts();
}

function clearBreath() {
  state.breath = null;
  clearPrediction();
  $("summarySource").textContent = "—";
  $("summaryBreath").textContent = "—";
  $("summarySamples").textContent = "—";
  $("summaryMechanics").textContent = "—";
  $("summaryBreath").textContent = "Loading dataset...";
  $("twinStatus").textContent = "WAITING FOR BREATH";
  $("predictButton").disabled = true;
  $("pressureEmpty").style.display = "block";
  $("inputEmpty").style.display = "block";
  drawCharts();
}

function setBusy(busy) {
  state.busy = busy;
  $("loadButton").disabled = busy;
  $("predictButton").disabled = busy || !state.breath;
  $("previousButton").disabled = busy;
  $("nextButton").disabled = busy;
  $("sourceSelect").disabled = busy;
  $("breathId").disabled = busy;
  $("modelSelect").disabled = busy;
  $("scenarioButton").disabled = busy || !(state.breath?.has_pressure && state.prediction?.model_version === "1.0.0");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  let payload;
  try { payload = await response.json(); }
  catch { throw new Error("The local app returned an unreadable response."); }
  if (!response.ok) throw new Error(payload.error || "The request could not be completed.");
  return payload;
}

async function prepareSource() {
  if (state.busy) return;
  clearBreath();
  const source = $("sourceSelect").value;
  $("sourceMeta").textContent = "Scanning the local file for breath boundaries…";
  $("loadMessage").textContent = "The first scan reads the CSV once to build a local breath index. The app does not load or copy the whole dataset into memory.";
  setStatus("Preparing local dataset index…", "busy");
  setBusy(true);
  try {
    const meta = await requestJson(`/api/metadata?source=${encodeURIComponent(source)}`);
    state.metadata = meta;
    const twin = await requestJson(`/api/twin-state?source=${encodeURIComponent(source)}`);
    if (twin.twin_state) showTwinState(twin.twin_state);
    else resetTwinDisplay("No valid breath processed in this dataset session.");
    $("summarySource").textContent = meta.file_name;
    $("summaryBreath").textContent = `${meta.first_breath_id} / ${meta.breath_count.toLocaleString()}`;
    $("sourceMeta").textContent = `${meta.breath_count.toLocaleString()} breaths · ${meta.row_count.toLocaleString()} rows`;
    $("breathId").value = String(meta.first_breath_id);
    $("loadMessage").textContent = `${meta.file_name} is ready. Breath IDs are indexed locally; measured pressure ${meta.has_pressure ? "is available" : "is not included"} in this file.`;
    setStatus("Dataset ready on this computer.", "success");
  } catch (error) {
    state.metadata = null;
    $("sourceMeta").textContent = "Dataset unavailable";
    $("loadMessage").textContent = error.message;
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function loadSelectedBreath() {
  if (state.busy) return;
  clearBreath();
  const source = $("sourceSelect").value;
  const breathId = $("breathId").value.trim();
  if (!/^-?\d+$/.test(breathId)) {
    setStatus("Enter a whole-number breath ID.", "error");
    return;
  }
  setBusy(true);
  setStatus(`Loading breath ${breathId}…`, "busy");
  $("loadMessage").textContent = "Reading only the selected breath from the local CSV.";
  try {
    const breath = await requestJson(`/api/breath?source=${encodeURIComponent(source)}&breath_id=${encodeURIComponent(breathId)}`);
    const condition = breath.rows.length ? [breath.rows[0].R, breath.rows[0].C] : null;
    if (state.twinCondition && condition && JSON.stringify(state.twinCondition) !== JSON.stringify(condition)) {
      await requestJson("/api/twin-reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source }) });
      resetTwinDisplay("Previous state cleared because the selected test-lung R/C condition changed.");
      state.twinCondition = null;
    }
    state.breath = breath;
    showBenchmarkCondition(condition, breath.breath_id, breath.rows.length === 80);
    $("summarySource").textContent = breath.file_name;
    $("summaryBreath").textContent = `${breath.breath_id} / ${state.metadata.breath_count.toLocaleString()}`;
    $("summarySamples").textContent = `${breath.rows.length} / 80`;
    const r = breath.rows[0].R;
    const c = breath.rows[0].C;
    $("summaryMechanics").textContent = r == null || c == null ? "Not supplied" : `${r} / ${c}`;
    $("pressureEmpty").style.display = "none";
    $("inputEmpty").style.display = "none";
    $("twinStatus").textContent = "BREATH LOADED · READY TO UPDATE";
    $("twinCoreNote").textContent = "Breath loaded. Generate a pressure prediction to update the fixed-condition replay state.";
    drawCharts();
    $("predictButton").disabled = false;
    const pressureMessage = breath.has_pressure ? "Measured pressure is shown as the reference curve." : "This test.csv breath has no measured pressure label.";
    $("loadMessage").textContent = pressureMessage;
    setStatus(`Breath ${breath.breath_id} loaded.`, "success");
  } catch (error) {
    state.breath = null;
    $("loadMessage").textContent = error.message;
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function navigateBreath(direction) {
  if (state.busy) return;
  const source = $("sourceSelect").value;
  const breathId = $("breathId").value.trim();
  if (!state.metadata) { await prepareSource(); return; }
  setBusy(true);
  try {
    const result = await requestJson(`/api/adjacent?source=${encodeURIComponent(source)}&breath_id=${encodeURIComponent(breathId)}&direction=${direction}`);
    $("breathId").value = result.breath_id;
  } catch (error) {
    setStatus(error.message, "error");
    setBusy(false);
    return;
  }
  setBusy(false);
  await loadSelectedBreath();
}

async function generatePrediction() {
  if (!state.breath || state.busy) return;
  clearPrediction();
  setBusy(true);
  setStatus("Generating prediction locally…", "busy");
  $("resultStatus").textContent = "Running";
  try {
    const result = await requestJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: state.breath.source,
        breath_id: state.breath.breath_id,
        model: $("modelSelect").value,
      }),
    });
    if (result.status !== "ok") {
      state.prediction = null;
      $("resultStatus").textContent = result.status === "unavailable" ? "Unavailable" : result.status;
      $("resultMessage").textContent = result.reason || result.error || "The predictor did not return a curve.";
      $("predictionDetails").hidden = false;
      $("predictionError").textContent = result.reason || result.error || "The predictor did not return a curve.";
      setStatus(result.reason || result.error || "No prediction curve was returned.", result.status === "unavailable" ? "neutral" : "error");
      drawCharts();
      return;
    }
    const expectedIds = state.breath.rows.map((row) => row.id);
    if (JSON.stringify(result.row_ids) !== JSON.stringify(expectedIds) || result.pressure.length !== expectedIds.length) {
      throw new Error("Prediction IDs or curve length do not match this breath. No curve was shown.");
    }
    if (!result.pressure.every((value) => Number.isFinite(value))) {
      throw new Error("The predictor returned an invalid pressure value. No curve was shown.");
    }
    state.prediction = result;
    if (result.twin_state) showTwinState(result.twin_state);
    $("predictionDetails").hidden = true;
    $("predictionError").textContent = result.reason || "";
    const peak = Math.max(...result.pressure);
    $("peakMetric").textContent = `${peak.toFixed(2)} cmH₂O`;
    const actual = state.breath.rows.map((row) => row.pressure);
    if (actual.every((value) => Number.isFinite(value))) {
      const mae = actual.reduce((sum, value, index) => sum + Math.abs(value - result.pressure[index]), 0) / actual.length;
      $("maeMetric").textContent = `${mae.toFixed(2)} cmH₂O`;
    } else {
      $("maeMetric").textContent = "Not available";
    }
    const hasUncertainty = Array.isArray(result.uncertainty) && result.uncertainty.some((value) => Number.isFinite(value));
    $("uncertaintyMetric").textContent = hasUncertainty ? "Provided by model" : "Not provided";
    $("scenarioButton").disabled = !(state.breath.has_pressure && result.model_version === "1.0.0");
    $("scenarioSummary").textContent = $("scenarioButton").disabled ? "Counterfactual requires train.csv and trained benchmark model." : "Ready to compare a modified input.";
    $("resultStatus").textContent = result.model_version?.startsWith("mock-") ? "Mock output" : "Prediction ready";
    $("resultMessage").textContent = result.reason || "Prediction completed. Review the displayed curve and its source before interpreting it.";
    $("twinStatus").textContent = result.twin_state?.state_validity?.startsWith("Valid") ? "ACTIVE · UPDATED" : "ACTIVE · BENCHMARK-CONDITIONED";
    $("twinCoreNote").textContent = result.twin_state?.state_validity || "Pressure prediction ready; no mechanics estimate is available.";
    setStatus(result.model_version?.startsWith("mock-") ? "Mock demonstration curve generated; it is not a trained model." : "Benchmark prediction generated.", "success");
    drawCharts();
    drawScenario();
  } catch (error) {
    state.prediction = null;
    $("resultStatus").textContent = "Error";
    $("resultMessage").textContent = error.message;
    $("predictionDetails").hidden = false;
    $("predictionError").textContent = error.message;
    $("peakMetric").textContent = "—";
    $("maeMetric").textContent = "—";
    $("uncertaintyMetric").textContent = "—";
    setStatus(error.message, "error");
    drawCharts();
  } finally {
    setBusy(false);
  }
}

function drawLine(canvas, rows, series, options = {}) {
  const ctx = canvas.getContext("2d");
  const box = canvas.getBoundingClientRect();
  const width = Math.max(250, box.width);
  const height = 245;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!rows || !rows.length) return;
  const margin = { left: 43, right: 12, top: 12, bottom: 30 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xs = rows.map((row) => row.time_step);
  const all = series.flatMap((item) => item.values.filter(Number.isFinite));
  if (!all.length) return;
  let minY = Math.min(...all);
  let maxY = Math.max(...all);
  if (options.backgroundValues) {
    minY = Math.min(minY, ...options.backgroundValues);
    maxY = Math.max(maxY, ...options.backgroundValues);
  }
  if (maxY === minY) { maxY += 1; minY -= 1; }
  const pad = (maxY - minY) * .08;
  minY -= pad; maxY += pad;
  const minX = xs[0];
  const maxX = xs[xs.length - 1] === minX ? minX + 1 : xs[xs.length - 1];
  const xPos = (x) => margin.left + ((x - minX) / (maxX - minX)) * plotW;
  const yPos = (y) => margin.top + (1 - (y - minY) / (maxY - minY)) * plotH;

  ctx.font = "10px Aptos, Segoe UI, sans-serif";
  ctx.strokeStyle = "#e7edf1";
  ctx.fillStyle = "#82919e";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = margin.top + (plotH * i) / 4;
    const value = maxY - ((maxY - minY) * i) / 4;
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(width - margin.right, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(value.toFixed(1), margin.left - 7, y);
  }
  ctx.strokeStyle = "#c8d4dc";
  ctx.beginPath(); ctx.moveTo(margin.left, margin.top + plotH); ctx.lineTo(width - margin.right, margin.top + plotH); ctx.stroke();
  ctx.fillStyle = "#82919e"; ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText(`${minX.toFixed(2)} s`, margin.left, margin.top + plotH + 8);
  ctx.fillText(`${maxX.toFixed(2)} s`, width - margin.right, margin.top + plotH + 8);

  if (options.valveMask) {
    const step = plotW / rows.length;
    ctx.fillStyle = "#f1e6dd";
    rows.forEach((row, i) => {
      if (row.u_out === 1) ctx.fillRect(xPos(row.time_step), margin.top, Math.max(1, step), plotH);
    });
    ctx.globalAlpha = 1;
  }
  for (const line of series) {
    ctx.beginPath();
    let started = false;
    line.values.forEach((value, i) => {
      if (!Number.isFinite(value)) { started = false; return; }
      const x = xPos(xs[i]); const y = yPos(value);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = line.color;
    ctx.lineWidth = line.width || 2;
    ctx.setLineDash(line.dash ? [5, 4] : []);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawCharts() {
  const rows = state.breath?.rows || [];
  const actual = rows.map((row) => row.pressure);
  const predicted = state.prediction?.pressure || [];
  drawLine($("pressureChart"), rows, [
    { values: actual, color: "#12304a", width: 2 },
    { values: predicted, color: "#e28e45", width: 2, dash: true },
  ]);
  drawLine($("inputChart"), rows, [
    { values: rows.map((row) => row.u_in), color: "#087e8b", width: 2 },
  ], { valveMask: true });
}

function showTwinState(twin) {
  state.twinCondition = twin.benchmark_condition || state.twinCondition;
  const [r, c] = twin.benchmark_condition || [];
  $("twinR").textContent = r == null ? "—" : String(r);
  $("twinC").textContent = c == null ? "—" : String(c);
  $("twinSource").textContent = r == null || c == null ? "Waiting" : "Benchmark test-lung condition";
  $("twinUncertainty").textContent = "Not available";
  $("twinTau").textContent = "Not available";
  $("twinCount").textContent = twin.valid_breaths || 0;
  $("twinBreath").textContent = twin.breath_id;
  $("twinQuality").textContent = twin.signal_quality;
  $("twinValidity").textContent = twin.state_validity?.startsWith("Valid") ? "VALID · replay summary updated" : "VALID · benchmark condition";
  $("twinConfidence").textContent = "R/C estimation from the supplied signals is not currently identifiable. Uncertainty: Not available.";
  $("twinUpdatedAt").textContent = twin.updated_at ? `Last updated ${new Date(twin.updated_at).toLocaleString()}.` : "";
  $("twinStatus").textContent = r == null || c == null ? "WAITING FOR BREATH" : (twin.state_validity?.startsWith("Valid") ? "ACTIVE · UPDATED" : "ACTIVE");
  $("twinCoreNote").textContent = r == null || c == null ? "Waiting for benchmark test-lung condition." : "Benchmark-conditioned test-lung mechanics · reference values used by the pressure model.";
}

function showBenchmarkCondition(condition, breathId, validSignal) {
  const [r, c] = condition || [];
  state.twinCondition = condition;
  $("twinR").textContent = r == null ? "—" : String(r);
  $("twinC").textContent = c == null ? "—" : String(c);
  $("twinSource").textContent = r == null || c == null ? "Unavailable" : "Benchmark test-lung condition";
  $("twinStatus").textContent = r == null || c == null ? "WAITING FOR BREATH" : "ACTIVE";
  $("twinCoreNote").textContent = r == null || c == null ? "Benchmark condition unavailable." : "Benchmark-conditioned test-lung mechanics · R/C are reference labels, not waveform estimates.";
  $("twinValidity").textContent = validSignal && r != null && c != null ? "VALID · benchmark condition" : "INVALID / unavailable";
  const finiteSignal = validSignal && state.breath?.rows?.every((row) => Number.isFinite(row.time_step) && Number.isFinite(row.u_in));
  $("twinQuality").textContent = finiteSignal ? "GOOD · 80 samples" : "INVALID / unavailable";
  $("twinBreath").textContent = breathId ?? "—";
  $("twinUncertainty").textContent = "Not available";
  $("twinTau").textContent = "Not available";
  $("twinConfidence").textContent = "R/C estimation from the supplied signals is not currently identifiable. Uncertainty: Not available.";
}

function drawScenario() {
  const rows = state.breath?.rows || [];
  const predicted = state.scenario?.current_pressure || state.prediction?.pressure || [];
  const proposed = state.scenario?.pressure || [];
  drawLine($("scenarioChart"), rows, [
    { values: predicted, color: "#e28e45", width: 2, dash: true },
    { values: proposed, color: "#087e8b", width: 2 },
  ]);
}

async function simulateScenario() {
  if (!state.breath || !state.prediction || state.busy) return;
  const scale = Number($("scenarioScale").value);
  if (!Number.isFinite(scale) || scale < 0 || scale > 2) { setStatus("Input scale must be between 0 and 2.", "error"); return; }
  setBusy(true); setStatus("Running local test-lung counterfactual…", "busy");
  try {
    state.scenario = null;
    $("scenarioSummary").textContent = "Running both current and proposed model inference…";
    $("scenarioState").textContent = "Simulating…";
    $("effectStatus").textContent = "Simulating…";
    $("currentPeakMetric").textContent = "—"; $("proposedPeakMetric").textContent = "—"; $("peakChangeMetric").textContent = "—";
    $("scenarioExplanation").textContent = "The model is generating current and proposed pressure curves…";
    drawScenario();
    const result = await requestJson("/api/scenario", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: state.breath.source, breath_id: state.breath.breath_id, model: "benchmark", u_in_scale: scale }) });
    if (result.status !== "ok") throw new Error(result.reason || result.error || "Scenario prediction unavailable.");
    state.scenario = result;
    state.prediction = { ...state.prediction, pressure: result.current_pressure };
    const currentPeak = Math.max(...result.current_pressure);
    const proposedPeak = Math.max(...result.pressure);
    const difference = result.peak_change;
    $("currentPeakMetric").textContent = `${currentPeak.toFixed(2)} cmH₂O`;
    $("proposedPeakMetric").textContent = `${proposedPeak.toFixed(2)} cmH₂O`;
    $("peakChangeMetric").textContent = `${difference >= 0 ? "+" : ""}${difference.toFixed(2)} cmH₂O`;
    $("effectStatus").textContent = "Scenario simulated";
    $("scenarioState").textContent = "Simulated";
    const effectText = result.overlap
      ? `The input waveform was scaled by ${scale.toFixed(2)}×. Current and PROPOSED predictions overlap, so the peak remains ${currentPeak.toFixed(2)} cmH₂O.`
      : `The input waveform was scaled by ${scale.toFixed(2)}× and passed through the pressure model. Predicted peak airway pressure ${difference >= 0 ? "increased" : "decreased"} by ${Math.abs(difference).toFixed(2)} cmH₂O.`;
    $("scenarioExplanation").textContent = effectText;
    $("scenarioSummary").textContent = result.overlap
      ? "Scenario simulated. CURRENT and PROPOSED predictions overlap exactly."
      : "Scenario simulated. See CURRENT vs PROPOSED response below.";
    setStatus("Benchmark counterfactual generated. Research simulation only.", "success"); drawScenario();
  } catch (error) { state.scenario = null; $("scenarioSummary").textContent = error.message; $("scenarioState").textContent = "Scenario simulation unavailable"; $("effectStatus").textContent = "Unavailable"; $("currentPeakMetric").textContent = "—"; $("proposedPeakMetric").textContent = "—"; $("peakChangeMetric").textContent = "—"; $("scenarioExplanation").textContent = error.message; setStatus(error.message, "error"); drawScenario(); }
  finally { setBusy(false); }
}

function pauseReplay() { if (state.replayTimer) clearInterval(state.replayTimer); state.replayTimer = null; }
function playReplay() {
  pauseReplay();
  const tick = async () => {
    if (!state.replayTimer) return;
    try {
      if (!state.busy && state.metadata) {
        const source = $("sourceSelect").value;
        const currentId = $("breathId").value;
        const next = await requestJson(`/api/adjacent?source=${encodeURIComponent(source)}&breath_id=${encodeURIComponent(currentId)}&direction=1`);
        if (String(next.breath_id) === currentId) { pauseReplay(); return; }
        $("breathId").value = next.breath_id; await loadSelectedBreath();
        if (state.breath && $("modelSelect").value === "benchmark") await generatePrediction();
      }
    } catch (error) { pauseReplay(); setStatus(`Replay stopped: ${error.message}`, "error"); return; }
    state.replayTimer = setTimeout(tick, Number($("replaySpeed").value));
  };
  state.replayTimer = setTimeout(tick, Number($("replaySpeed").value));
}

function resetTwinDisplay(message) {
  ["twinR", "twinC"].forEach((id) => $(id).textContent = "—");
  $("twinTau").textContent = "Not available";
  $("twinSource").textContent = "Waiting"; $("twinUncertainty").textContent = "Not available";
  $("twinCount").textContent = "0"; $("twinBreath").textContent = "—";
  $("twinQuality").textContent = "Waiting"; $("twinConfidence").textContent = `R/C estimation from the supplied signals is not currently identifiable. Uncertainty: Not available. ${message}`;
  $("twinValidity").textContent = "State not yet updated";
  $("twinStatus").textContent = "WAITING FOR BREATH";
  $("twinCoreNote").textContent = message;
  $("twinUpdatedAt").textContent = "";
}

async function clearLocalData() {
  if (!window.confirm("Clear the generated breath index and local app event log? The source CSV files will not be changed.")) return;
  try {
    const result = await requestJson("/api/clear-local", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    clearBreath();
    state.metadata = null;
    $("sourceMeta").textContent = "Local index cleared";
    $("loadMessage").textContent = result.message;
    setStatus("Local index and app event log cleared.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

$("sourceSelect").addEventListener("change", () => { state.metadata = null; prepareSource(); });
$("loadButton").addEventListener("click", loadSelectedBreath);
$("predictButton").addEventListener("click", generatePrediction);
$("previousButton").addEventListener("click", () => navigateBreath(-1));
$("nextButton").addEventListener("click", () => navigateBreath(1));
$("breathId").addEventListener("keydown", (event) => { if (event.key === "Enter") loadSelectedBreath(); });
$("clearButton").addEventListener("click", clearLocalData);
$("scenarioButton").addEventListener("click", simulateScenario);
$("scenarioScale").addEventListener("input", () => {
  state.scenario = null;
  $("scenarioSummary").textContent = "Scale changed. Simulate again to generate a matching prediction.";
  $("scenarioState").textContent = "Simulation ready"; $("effectStatus").textContent = "Run a scenario to compare";
  $("currentPeakMetric").textContent = "—"; $("proposedPeakMetric").textContent = "—"; $("peakChangeMetric").textContent = "—";
  $("scenarioExplanation").textContent = "Scale changed. Run the scenario again to update the comparison.";
  drawScenario();
});
$("playButton").addEventListener("click", playReplay);
$("pauseButton").addEventListener("click", pauseReplay);
$("resetTwinButton").addEventListener("click", async () => {
  pauseReplay(); await requestJson("/api/twin-reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source: $("sourceSelect").value }) });
  state.twinCondition = null;
  resetTwinDisplay("Twin state reset.");
});
window.addEventListener("resize", () => { drawCharts(); drawScenario(); });

requestJson("/api/health").then(async () => {
  const model = await requestJson("/api/model-status");
  $("modelReadyPill").textContent = model.status === "ready" ? `● Model ready · ${model.model_version}` : "● Model unavailable";
  if (model.status !== "ready") $("modelReadyPill").title = model.reason || "Model unavailable";
  setStatus("Local app ready. Preparing dataset index…", "busy");
  return prepareSource();
}).catch((error) => setStatus(error.message, "error"));
