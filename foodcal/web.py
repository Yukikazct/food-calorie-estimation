"""Local food-classification and portion-calorie demo, with honest readiness."""
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit

from .data import load_config


class FoodApp:
    def __init__(self, config="configs/default.json", checkpoint="models/food10/best.pt", device="auto"):
        self.config = load_config(config)
        self.checkpoint = Path(checkpoint)
        self.model = None
        self.metadata = None
        self.device = device
        self.lock = threading.Lock()
        self.load_error = None
        if self.checkpoint.is_file():
            self._load_checkpoint()

    def _load_checkpoint(self):
        with self.lock:
            if self.model is not None:
                return
            from .model import load_classifier, resolve_device, image_transform
            device = resolve_device(self.device)
            model, metadata = load_classifier(self.checkpoint, device)
            if metadata["class_names"] != self.config["classes"]:
                raise ValueError("Food checkpoint classes do not match the configured task")
            self.transform = image_transform(metadata["input_size"], augment=False)
            self.device, self.metadata, self.model = device, metadata, model
            self.load_error = None

    def info(self):
        if self.model is None and self.checkpoint.is_file():
            try:
                self._load_checkpoint()
            except (ValueError, OSError, RuntimeError, KeyError):
                self.load_error = "菜品模型文件未能加载，请检查训练结果；手动热量估算仍可使用。"
        nutrition_path = Path(self.config["nutrition"]["path"])
        entries = {}
        if nutrition_path.exists():
            from .nutrition import load_nutrition
            entries = load_nutrition(nutrition_path).get("entries", {})
        download = None
        status_path = Path("runs/setup/food101_download.json")
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
                download = {k: status[k] for k in ("state", "bytes", "total_bytes", "percent", "selected_images") if k in status}
            except (ValueError, OSError):
                pass
        return {"app": "food-calorie", "model_ready": self.model is not None,
                "classes": [{"id": c, "name": n, "nutrition_ready": entries.get(c, {}).get("status") == "verified_reference"}
                            for c, n in zip(self.config["classes"], self.config["class_names_zh"])],
                "data_preparation": download,
                "model_message": "菜品模型已就绪" if self.model is not None else (self.load_error or "菜品模型尚未训练；可以先手选菜品，按输入份量估算热量。")}

    def estimate(self, payload):
        from .nutrition import load_nutrition, estimate_calories
        if payload.get("class_name") not in self.config["classes"]:
            raise ValueError("请选择支持的菜品")
        if type(payload.get("grams")) not in (int, float):
            raise ValueError("请输入以克为单位的份量")
        return estimate_calories(load_nutrition(self.config["nutrition"]["path"]), payload["class_name"], payload["grams"])

    def predict(self, payload):
        if self.model is None:
            raise RuntimeError("菜品模型尚未训练，当前不生成识别结果")
        from PIL import Image, ImageOps
        import torch
        image_bytes = base64.b64decode(payload["image"], validate=True)
        with Image.open(io.BytesIO(image_bytes)) as original:
            if original.width * original.height > 20_000_000 or min(original.size) <= 0:
                raise ValueError("图片尺寸过大")
            rgb = ImageOps.exif_transpose(original).convert("RGB")
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with self.lock, torch.inference_mode():
            values, indices = self.model(tensor).softmax(1)[0].topk(3)
        return {"predictions": [{"class_name": self.config["classes"][int(index)],
                                  "name_zh": self.config["class_names_zh"][int(index)],
                                  "probability": float(value)} for value, index in zip(values.cpu(), indices.cpu())],
                "note": "分类结果请核对；重量由你输入，不能从照片确定。"}


def create_server(app, port=8780):
    token = secrets.token_urlsafe(32)
    static = Path(__file__).parent / "static"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def local(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def reply(self, status, value, kind="application/json; charset=utf-8"):
            data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.local():
                return self.reply(403, {"error": "Local access only"})
            path = urlsplit(self.path).path
            if path == "/api/info":
                return self.reply(200, {**app.info(), "token": token})
            if path in ("/", "/app.js"):
                name = "index.html" if path == "/" else "app.js"
                kind = "text/html; charset=utf-8" if path == "/" else "text/javascript; charset=utf-8"
                return self.reply(200, (static / name).read_bytes(), kind)
            return self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.local() or not secrets.compare_digest(self.headers.get("X-Food-Token", ""), token):
                return self.reply(403, {"error": "Invalid local session"})
            route = urlsplit(self.path).path
            if route not in ("/api/predict", "/api/estimate"):
                return self.reply(404, {"error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 12_000_000:
                    raise ValueError("请求大小不符合要求")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Invalid request")
                result = app.predict(payload) if route == "/api/predict" else app.estimate(payload)
                return self.reply(200, result)
            except RuntimeError as exc:
                return self.reply(503, {"error": str(exc)})
            except (ValueError, TypeError, KeyError, OSError) as exc:
                return self.reply(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
