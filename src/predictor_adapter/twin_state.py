"""Fixed-condition benchmark replay summaries, never inferred clinical R/C.

The simulator R/C labels identify replay compatibility/evaluation conditions only.
Because u_in is not calibrated flow and no volume signal is supplied, R/C and a
physical time constant are not identifiable from these data. The persistent
session state therefore tracks directly measurable waveform summaries instead.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import threading
from typing import Any


class TwinStateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _summarize(rows: list[dict[str, Any]]) -> dict[str, float] | None:
        if len(rows) != 80 or any(row.get("pressure") is None for row in rows):
            return None
        times = [float(row["time_step"]) for row in rows]
        pressure = [float(row["pressure"]) for row in rows]
        flow_input = [float(row["u_in"]) for row in rows]
        if not all(math.isfinite(v) for v in times + pressure + flow_input):
            return None
        duration = times[-1] - times[0]
        if duration <= 0:
            return None
        # Direct, identifiable descriptive features. No simulator R/C labels enter.
        return {"peak_pressure": max(pressure), "mean_pressure": sum(pressure) / len(pressure),
                "input_peak": max(flow_input),
                "pressure_change_rate": (pressure[-1] - pressure[0]) / duration}

    def update(self, session: str, rows: list[dict[str, Any]], breath_id: int, model_version: str) -> dict[str, Any]:
        summary = self._summarize(rows)
        # R/C values are used only to prevent pooling incompatible simulator conditions.
        condition = (rows[0].get("R"), rows[0].get("C")) if rows else (None, None)
        with self._lock:
            previous = self._states.get(session)
            reset_for_condition = previous is not None and previous.get("benchmark_condition") != condition
            compatible_previous = None if reset_for_condition else previous
            processed_ids = list(compatible_previous.get("processed_breath_ids", [])) if compatible_previous else []
            if summary is None:
                state = {**(compatible_previous or {}), "breath_id": breath_id,
                         "signal_quality": "INVALID — requires 80 finite samples and measured pressure",
                         "state_validity": "Not updated", "updated_at": datetime.now(timezone.utc).isoformat(),
                         "model_version": model_version}
            else:
                if compatible_previous and breath_id in processed_ids:
                    return dict(compatible_previous)
                processed_ids.append(breath_id)
                count = len(processed_ids)
                means = {}
                for key, value in summary.items():
                    old = compatible_previous.get("waveform_summary", {}).get(key) if compatible_previous else None
                    means[key] = value if old is None else old + (value - old) / count
                state = {"estimated_resistance": None, "estimated_compliance": None,
                         "time_constant": None,
                         "mechanics_status": "Not identifiable: benchmark lacks calibrated flow and volume",
                         "waveform_summary": means, "last_breath_summary": summary,
                         "benchmark_condition": condition,
                         "valid_breaths": count, "processed_breath_ids": processed_ids,
                         "breath_id": breath_id,
                         "signal_quality": "GOOD — 80 finite measured samples",
                         "state_uncertainty": "R/C uncertainty unavailable because R/C are not identifiable from supplied signals",
                         "state_validity": "Valid fixed-condition test-lung replay summary",
                         "updated_at": datetime.now(timezone.utc).isoformat(), "model_version": model_version}
            self._states[session] = state
            return dict(state)

    def reset(self, session: str) -> None:
        with self._lock:
            self._states.pop(session, None)

    def snapshot(self, session: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._states.get(session)
            return dict(value) if value else None


TWIN_STATES = TwinStateStore()
