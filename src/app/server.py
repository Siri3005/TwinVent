"""Loopback-only HTTP server for the TwinVent offline benchmark replay."""

from __future__ import annotations

import json
import math
import mimetypes
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
WEB = Path(__file__).resolve().parent / "web"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_io.dataset_store import DatasetError, STORE  # noqa: E402
from local_storage.session_log import clear_local_state, log_event  # noqa: E402
from predictor_adapter.benchmark import BenchmarkModelAdapter  # noqa: E402
from predictor_adapter.contracts import (  # noqa: E402
    PredictionError,
    PredictionResult,
    PredictorUnavailable,
    validate_prediction,
)
from predictor_adapter.mock import MockPredictor  # noqa: E402


MAX_REQUEST_BYTES = 16_384
PREDICTION_TIMEOUT_SECONDS = 5
PREDICTION_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="twinvent-predict")
STATIC_FILES = {
    "/assets/app.js": WEB / "app.js",
    "/assets/styles.css": WEB / "styles.css",
}


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "TwinVentReplay/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid writing breath data or request contents to the terminal.
        print(f"[TwinVent] {self.address_string()} - {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json_bytes(payload))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            page = WEB / "index.html"
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return
        static = STATIC_FILES.get(parsed.path)
        if static is not None:
            mime = mimetypes.guess_type(static.name)[0] or "application/octet-stream"
            self._send(200, static.read_bytes(), f"{mime}; charset=utf-8")
            return
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self._json(200, {"status": "ok", "offline": True, "app_version": "0.1.0"})
            elif parsed.path == "/api/metadata":
                source = query.get("source", [""])[0]
                metadata = STORE.metadata(source)
                log_event("metadata_loaded", source=source, status="ok")
                self._json(200, metadata)
            elif parsed.path == "/api/breath":
                source = query.get("source", [""])[0]
                breath_id = query.get("breath_id", [""])[0]
                breath = STORE.load_breath(source, breath_id)
                log_event("breath_loaded", source=source, breath_id=breath["breath_id"], status="ok")
                self._json(200, breath)
            elif parsed.path == "/api/adjacent":
                source = query.get("source", [""])[0]
                breath_id = query.get("breath_id", [""])[0]
                direction = int(query.get("direction", ["1"])[0])
                if direction not in (-1, 1):
                    raise DatasetError("Navigation direction must be -1 or 1.")
                self._json(200, {"breath_id": STORE.adjacent_id(source, breath_id, direction)})
            else:
                self._json(404, {"error": "This local app route was not found."})
        except DatasetError as exc:
            self._json(400, {"error": str(exc)})
        except (ValueError, OSError) as exc:
            self._json(400, {"error": f"The request could not be completed: {exc}"})
        except Exception:
            self._json(500, {"error": "The local app encountered an unexpected error. No prediction was shown."})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/predict" and self.path != "/api/clear-local":
            self._json(404, {"error": "This local app route was not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request is too large.")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object.")
            if self.path == "/api/clear-local":
                STORE.clear_cached_indexes()
                clear_local_state()
                self._json(200, {"status": "cleared", "message": "Local index cache and app event log cleared. Source CSV files were not changed."})
                return
            source = str(payload.get("source", ""))
            breath_id = str(payload.get("breath_id", ""))
            model = str(payload.get("model", "mock"))
            breath = STORE.load_breath(source, breath_id)
            rows = breath["rows"]
            if model == "mock":
                predictor = MockPredictor()
            elif model == "benchmark":
                predictor = BenchmarkModelAdapter()
            else:
                raise ValueError("Select mock or benchmark as the predictor.")
            # Keep measured pressure out of the model input to prevent label leakage.
            features = [
                {key: row.get(key) for key in ("id", "time_step", "u_in", "u_out", "R", "C")}
                for row in rows
            ]
            future = PREDICTION_POOL.submit(predictor.predict_breath, features)
            try:
                result = future.result(timeout=PREDICTION_TIMEOUT_SECONDS)
            except FutureTimeout:
                future.cancel()
                result = PredictionResult(
                    row_ids=[int(row["id"]) for row in features],
                    pressure=[],
                    uncertainty=[],
                    status="error",
                    reason=f"Predictor did not respond within {PREDICTION_TIMEOUT_SECONDS} seconds. No curve was shown.",
                    model_version=getattr(predictor, "model_version", model),
                )
            validate_prediction(result, features)
            log_event("prediction", source=source, breath_id=breath["breath_id"], status=result.status, model_version=result.model_version)
            self._json(200, {**result.as_dict(), "source": source, "breath_id": breath["breath_id"]})
        except PredictorUnavailable as exc:
            result = PredictionResult([], [], [], "unavailable", str(exc), "benchmark-unavailable")
            log_event("prediction", status="unavailable", model_version="benchmark-unavailable")
            self._json(200, result.as_dict())
        except (DatasetError, PredictionError, ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"status": "error", "error": str(exc)})
        except Exception:
            self._json(500, {"status": "error", "error": "Prediction failed. No curve was shown."})


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    if host != "127.0.0.1":
        raise ValueError("The demo server must stay bound to loopback (127.0.0.1).")
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print("TwinVent Offline Replay is ready.")
    print(f"Open http://{host}:{port} in a browser on this computer.")
    print("This is a test-lung benchmark demonstration. It does not control a ventilator.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the local replay app.")
    finally:
        httpd.server_close()
