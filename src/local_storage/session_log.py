"""Store only local app events, never waveform values."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / ".twinvent"
LOG_PATH = STATE_DIR / "logs" / "session.jsonl"
_LOCK = threading.Lock()


def log_event(event: str, **fields: Any) -> None:
    safe = {key: value for key, value in fields.items() if key in {"source", "breath_id", "status", "model_version"}}
    record = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **safe}
    with _LOCK:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def log_scenario(**fields: Any) -> None:
    """Locally audit scenario summaries without storing identifiers or waveforms."""
    allowed = {key: fields[key] for key in (
        "breath_id", "twin_state", "input_summary", "proposed_scenario",
        "predicted_output", "uncertainty", "model_version", "validity", "user_action"
    ) if key in fields}
    record = {"at": datetime.now(timezone.utc).isoformat(), "event": "counterfactual_simulation",
              "preprocessing_version": "benchmark-features-v1", **allowed}
    with _LOCK:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def clear_local_state() -> None:
    """Clear generated indexes and local event log, never source CSVs."""
    with _LOCK:
        cache = STATE_DIR / "cache"
        if cache.exists():
            for item in cache.iterdir():
                if item.is_file() and item.suffix in {".json", ".tmp"}:
                    item.unlink()
        if LOG_PATH.exists():
            LOG_PATH.unlink()

