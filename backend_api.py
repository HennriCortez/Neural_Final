import base64
import io
import os

import requests
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

from face_crop import crop_to_face  # face detection + crop


# ── Configuration ─────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_DIR, "best_vit_mask.pth")
MODEL_URL   = os.getenv("MODEL_URL")  # set this in Railway environment variables
CLASS_NAMES = ["With Mask", "Without Mask"]
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _download_model() -> None:
    """Download model weights if not present locally."""
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL:
        raise RuntimeError(
            "Model file not found and MODEL_URL env var is not set. "
            "Upload best_vit_mask.pth to a file host and set MODEL_URL."
        )
    print(f"Downloading model weights from remote …")
    hf_token = os.getenv("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    with requests.get(MODEL_URL, stream=True, timeout=300, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="\r")
    print("\nModel download complete.")


# ── Model ─────────────────────────────────────────────────────────────────────
class PartialViT(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        from transformers import ViTConfig
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        config.num_labels = num_classes
        self.vit = ViTForImageClassification(config)
        hidden = config.hidden_size  # 768
        self.vit.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        return self.vit(pixel_values=pixel_values).logits


# ── Inference transform ───────────────────────────────────────────────────────
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Load weights ──────────────────────────────────────────────────────────────
_download_model()
print(f"Loading model on {DEVICE} …")
model = PartialViT(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.eval()
print("Model loaded successfully.")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=_DIR)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        # 1. Detect face and crop
        img, face_found = crop_to_face(img)

        # 2. Encode cropped image as base64 (in-memory only, never saved)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        cropped_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 3. Classify
        tensor = val_transforms(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()

        predictions = [
            {
                "class":        CLASS_NAMES[i],
                "confidence":   round(probs[i], 4),
                "face_detected": face_found,
            }
            for i in range(len(CLASS_NAMES))
        ]
        predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return jsonify({
            "predictions":   predictions,
            "cropped_image": f"data:image/jpeg;base64,{cropped_b64}",
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
