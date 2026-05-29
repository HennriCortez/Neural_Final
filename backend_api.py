import base64
import io
import os

import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

from face_crop import crop_to_face  # face detection + crop


import os
import urllib.request

MODEL_PATH = "best_vit_mask.pth"
# TODO: Replace this with your actual direct download link from Step 1
MODEL_URL = "https://huggingface.co/your-username/your-repo/resolve/main/best_vit_mask.pth"

if not os.path.exists(MODEL_PATH):
    print(f"Downloading model weights from {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete successfully!")
    except Exception as e:
        print(f"Failed to download weights: {e}")
        raise e


# ── Configuration ─────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_DIR, "best_vit_mask.pth")
CLASS_NAMES = ["With Mask", "Without Mask"]
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────
class PartialViT(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        hidden = self.vit.config.hidden_size  # 768
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
print(f"Loading model on {DEVICE} …")
model = PartialViT(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully.")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
