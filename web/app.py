import os
import sys
import uuid
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web.config import ALLOWED_EXT, CHECKPOINT_PATH, MAX_UPLOAD_MB, UPLOAD_DIR
from web.model.inference import load_model, predict_video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_model_loaded = False


def ensure_model():
    global _model_loaded
    if not _model_loaded:
        load_model(CHECKPOINT_PATH)
        _model_loaded = True


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


@app.route("/health")
def health():
    ckpt_ok = os.path.exists(CHECKPOINT_PATH)
    import torch
    return {
        "status": "ok",
        "checkpoint": ckpt_ok,
        "cuda": torch.cuda.is_available(),
        "model_loaded": _model_loaded,
    }


@app.route("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_MB)


@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        return render_template("index.html", error="未选择文件", max_mb=MAX_UPLOAD_MB)

    file = request.files["video"]
    if not file.filename:
        return render_template("index.html", error="未选择文件", max_mb=MAX_UPLOAD_MB)

    if not allowed_file(file.filename):
        return render_template("index.html", error="仅支持 MP4 格式", max_mb=MAX_UPLOAD_MB)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    file.save(save_path)

    try:
        ensure_model()
        result = predict_video(save_path)
    except FileNotFoundError as e:
        return render_template("index.html", error=str(e), max_mb=MAX_UPLOAD_MB)
    except Exception as e:
        return render_template("index.html", error=f"推理失败: {e}", max_mb=MAX_UPLOAD_MB)
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    return render_template("result.html", result=result, filename=safe_name)


if __name__ == "__main__":
    ensure_model()
    app.run(host="0.0.0.0", port=6006, debug=False)
