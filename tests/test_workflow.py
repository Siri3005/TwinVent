import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app.server import Handler
from data_io.dataset_store import STORE


class TwinVentWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_port}"
        cls.breath_id = STORE.metadata("train")["first_breath_id"]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.thread.join(timeout=3)
        cls.httpd.server_close()

    def request(self, path, payload=None):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def test_prediction_updates_valid_fixed_condition_state(self):
        result = self.request("/api/predict", {"source": "train", "breath_id": str(self.breath_id), "model": "benchmark"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["pressure"]), 80)
        twin = result["twin_state"]
        self.assertEqual(twin["valid_breaths"], 1)
        self.assertIsNone(twin["estimated_resistance"])
        self.assertIn("identifiable", twin["mechanics_status"].lower())

    def test_identity_and_nonidentity_counterfactuals_use_model_outputs(self):
        payload = {"source": "train", "breath_id": str(self.breath_id), "model": "benchmark", "u_in_scale": 1.0}
        identity = self.request("/api/scenario", payload)
        self.assertEqual(identity["status"], "ok")
        self.assertTrue(identity["overlap"])
        self.assertEqual(identity["current_model_version"], identity["model_version"])
        self.assertEqual(identity["current_pressure"], identity["pressure"])

        payload["u_in_scale"] = 0.8
        changed = self.request("/api/scenario", payload)
        self.assertEqual(changed["status"], "ok")
        self.assertEqual(len(changed["current_pressure"]), 80)
        self.assertEqual(len(changed["pressure"]), 80)
        self.assertTrue(any(abs(a-b) > 1e-8 for a,b in zip(changed["current_pressure"], changed["pressure"])))
        self.assertAlmostEqual(changed["peak_change"], max(changed["pressure"])-max(changed["current_pressure"]), places=10)

    def test_counterfactual_rejects_unsupported_scale_and_test_labels(self):
        payload = {"source": "train", "breath_id": str(self.breath_id), "model": "benchmark", "u_in_scale": 2.1}
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/scenario", payload)
        self.assertEqual(caught.exception.code, 400)
        payload.update({"source": "test", "u_in_scale": 1.0})
        test_id = STORE.metadata("test")["first_breath_id"]
        payload["breath_id"] = str(test_id)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/scenario", payload)
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
