"""
face_crop.py
------------
Detects the largest/highest-confidence face in an image and returns a
padded crop centred on that face.

Detector priority:
  1. YuNet  (OpenCV 4.8+ built-in neural detector — handles masked faces)
  2. Haar cascade  (built-in fallback, no extra files needed)

Usage as a module
-----------------
    from face_crop import crop_to_face
    from PIL import Image

    img          = Image.open("photo.jpg").convert("RGB")
    cropped, ok  = crop_to_face(img)          # ok=False → full image returned
    cropped.save("cropped.jpg")

Usage from the command line
---------------------------
    python face_crop.py photo.jpg             # saves  photo_cropped.jpg
    python face_crop.py photo.jpg out.jpg     # saves  out.jpg
"""

import os
import sys
import urllib.request

import cv2
import numpy as np
from PIL import Image


# ── YuNet model ───────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(os.path.abspath(__file__))
_YUNET_URL  = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_PATH = os.path.join(_DIR, "yunet_face.onnx")


def _ensure_yunet():
    """Download the YuNet ONNX weight file if it is not already present."""
    if os.path.exists(_YUNET_PATH) and os.path.getsize(_YUNET_PATH) >= 100_000:
        return
    print("Downloading YuNet face model (~234 KB) …")
    req = urllib.request.Request(_YUNET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(_YUNET_PATH, "wb") as f:
        f.write(resp.read())
    print(f"  Saved to {_YUNET_PATH}  ({os.path.getsize(_YUNET_PATH):,} bytes)")


_ensure_yunet()

_yunet = cv2.FaceDetectorYN.create(
    model=_YUNET_PATH,
    config="",
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
)

# Haar cascade — zero-dependency fallback
_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Public API ────────────────────────────────────────────────────────────────

def crop_to_face(
    pil_img: Image.Image,
    padding: float = 0.30,
) -> tuple:
    """
    Detect the dominant face in *pil_img* and return a padded crop.

    Parameters
    ----------
    pil_img : PIL.Image.Image  (RGB)
        Input image.
    padding : float
        Fractional padding added around the detected bounding box on each side.
        0.30 means 30 % of the box width/height is added as a border.

    Returns
    -------
    (cropped : PIL.Image.Image, face_found : bool)
        *face_found* is False when no face was detected and the original
        image is returned unchanged.
    """
    img_np  = np.array(pil_img)
    h, w    = img_np.shape[:2]
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    x1 = y1 = x2 = y2 = None

    # ── 1. YuNet ──────────────────────────────────────────────────────────────
    _yunet.setInputSize((w, h))
    _, faces = _yunet.detect(img_bgr)   # Nx15 float32 array or None

    if faces is not None and len(faces) > 0:
        best = faces[np.argmax(faces[:, 14])]          # highest confidence
        bx, by, bw, bh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        print(f"[face_crop] YuNet  — score={best[14]:.3f}  box=({x1},{y1},{x2},{y2})")

    # ── 2. Haar cascade fallback ──────────────────────────────────────────────
    if x1 is None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        dets = _haar.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )
        if len(dets) > 0:
            bx, by, bw, bh = max(dets, key=lambda r: r[2] * r[3])
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh
            print(f"[face_crop] Haar   — box=({x1},{y1},{x2},{y2})")

    # ── No face detected ──────────────────────────────────────────────────────
    if x1 is None:
        print("[face_crop] No face detected — returning full image.")
        return pil_img, False

    # ── Apply padding and clamp to image bounds ───────────────────────────────
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return pil_img.crop((x1, y1, x2, y2)), True


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _cli():
    if len(sys.argv) < 2:
        print("Usage: python face_crop.py <input_image> [output_image]")
        sys.exit(1)

    src = sys.argv[1]
    if len(sys.argv) >= 3:
        dst = sys.argv[2]
    else:
        base, ext = os.path.splitext(src)
        dst = f"{base}_cropped{ext}"

    img = Image.open(src).convert("RGB")
    cropped, found = crop_to_face(img)
    cropped.save(dst)
    status = "face crop" if found else "full image (no face detected)"
    print(f"[face_crop] Saved {status} → {dst}")


if __name__ == "__main__":
    _cli()
