"""Local API readiness and input boundary checks, without fabricated predictions."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock
import urllib.error
import urllib.request

from foodcal.web import FoodApp, create_server


class FoodWebTests(unittest.TestCase):
    def test_model_becomes_available_after_atomic_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trained.pt"
            app = FoodApp(checkpoint=checkpoint, device="cpu")
            self.assertFalse(app.info()["model_ready"])
            checkpoint.write_bytes(b"mock checkpoint for lifecycle test only")
            metadata = {"class_names": app.config["classes"], "input_size": 224}
            with patch('foodcal.model.load_classifier', return_value=(MagicMock(), metadata)) as loader:
                self.assertTrue(app.info()["model_ready"])
                self.assertTrue(app.info()["model_ready"])
                loader.assert_called_once()

    def test_missing_checkpoint_has_no_fake_food_predictions(self):
        app = FoodApp(checkpoint="models/nonexistent-food-test-checkpoint.pt")
        self.assertIsNone(app.model)
        self.assertFalse(app.info()["model_ready"])
        with self.assertRaisesRegex(RuntimeError, "尚未训练"):
            app.predict({"image": ""})
        with self.assertRaises(ValueError):
            app.estimate({"class_name": "pizza", "grams": True})

    def test_local_api_serves_page_requires_token_and_protects_paths(self):
        app = FoodApp(checkpoint="models/nonexistent-food-test-checkpoint.pt")
        server = create_server(app, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(url + "/api/info") as response:
                info = json.load(response)
            self.assertEqual(info["app"], "food-calorie")
            for path in ("/", "/app.js"):
                with urllib.request.urlopen(url + path) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read())
            for request, expected in (
                (urllib.request.Request(url + "/api/info", headers={"Host": "outside.invalid"}), 403),
                (urllib.request.Request(url + "/api/estimate", data=b"{}"), 403),
                (urllib.request.Request(url + "/../../README.md"), 404),
                (urllib.request.Request(url + "/api/predict", data=b"{}", headers={"X-Food-Token": info["token"]}), 503),
                (urllib.request.Request(url + "/api/estimate", data=b'[]', headers={"X-Food-Token": info["token"]}), 400),
            ):
                with self.subTest(expected=expected), self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request)
                self.assertEqual(caught.exception.code, expected)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
