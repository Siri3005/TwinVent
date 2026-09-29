"use strict";

const $ = (id) => document.getElementById(id);
const state = { metadata: null, breath: null, prediction: null, busy: false };

function setStatus(message, kind = "neutral") {
  const badge = $("statusBadge");
  badge.className = `status-badge status-${kind}`;
  $("statusText").textContent = message;
}

function clearPrediction() {
  state.prediction = null;
  $("modelVersion").textContent = "";
  $("resultStatus").textContent = "Not run";
  $("resultMessage").textContent = "Choose a breath and generate a prediction.";
  $("peakMetric").textContent = "—";
  $("maeMetric").textContent = "—";
  $("uncertaintyMetric").textContent = "—";
  drawCharts();
}

function clearBreath() {
  state.breath = null;
  clearPrediction();
  $("summarySource").textContent = "—";
  $("summaryBreath").textContent = "—";
  $("summarySamples").textContent = "—";
  $("summaryMechanics").textContent = "—";
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
    state.breath = breath;
    $("summarySource").textContent = breath.file_name;
    $("summaryBreath").textContent = String(breath.breath_id);
    $("summarySamples").textContent = `${breath.rows.length} / 80`;
    const r = breath.rows[0].R;
    const c = breath.rows[0].C;
    $("summaryMechanics").textContent = r == null || c == null ? "Not supplied" : `${r} / ${c}`;
    $("pressureEmpty").style.display = "none";
    $("inputEmpty").style.display = "none";
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
    $("modelVersion").textContent = `Model ${result.model_version || "unknown"}`;
    $("resultStatus").textContent = result.model_version?.startsWith("mock-") ? "Mock output" : "Prediction ready";
    $("resultMessage").textContent = result.reason || "Prediction completed. Review the displayed curve and its source before interpreting it.";
    setStatus(result.model_version?.startsWith("mock-") ? "Mock demonstration curve generated; it is not a trained model." : "Benchmark prediction generated.", "success");
    drawCharts();
  } catch (error) {
    state.prediction = null;
    $("resultStatus").textContent = "Error";
    $("resultMessage").textContent = error.message;
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
window.addEventListener("resize", drawCharts);

requestJson("/api/health").then(() => {
  setStatus("Local app ready. Preparing dataset index…", "busy");
  return prepareSource();
}).catch((error) => setStatus(error.message, "error"));
