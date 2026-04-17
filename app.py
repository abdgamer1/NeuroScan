import os
import io
import base64
import tempfile
import numpy as np
from flask import Flask, request, jsonify, send_file

import nibabel as nib
from PIL import Image

import torch
import torchvision.transforms as transforms
import timm

import tensorflow as tf

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma"]
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PTH_PATH       = os.path.join(BASE_DIR, "classification.pth")
H5_PATH        = os.path.join(BASE_DIR, "finetuned_meningioma_model.h5")


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
def load_pytorch_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    state = torch.load(PTH_PATH, map_location=DEVICE)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def load_keras_model():
    return tf.keras.models.load_model(H5_PATH)

print("Loading models...")
try:
    clf_model = load_pytorch_model()
    print("✅ classification.pth loaded")
except Exception as e:
    clf_model = None
    print(f"❌ classification.pth failed: {e}")

try:
    seg_model = load_keras_model()
    print("✅ finetuned_meningioma_model.h5 loaded")
except Exception as e:
    seg_model = None
    print(f"❌ finetuned_meningioma_model.h5 failed: {e}")


# ─────────────────────────────────────────────
# NIFTI HELPERS
# ─────────────────────────────────────────────
def load_nifti_volume(file_storage) -> np.ndarray:
    """Save uploaded file to temp, load with nibabel, return numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name
    vol = nib.load(tmp_path).get_fdata()
    os.unlink(tmp_path)
    return vol.astype(np.float32)

def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return (vol - vmin) / (vmax - vmin)

def get_middle_slice(vol: np.ndarray) -> np.ndarray:
    """Return the axial middle slice (H x W)."""
    z = vol.shape[2] // 2
    return vol[:, :, z]

def slice_to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    """Convert a normalised 2-D float slice → uint8 RGB (H, W, 3)."""
    s = np.clip(slice_2d, 0, 1)
    uint8 = (s * 255).astype(np.uint8)
    return np.stack([uint8, uint8, uint8], axis=-1)

def volume_to_nifti_bytes(vol: np.ndarray) -> bytes:
    """Return a NIfTI file as bytes (no affine needed for mask)."""
    nii = nib.Nifti1Image(vol, affine=np.eye(4))
    buf = io.BytesIO()
    nii.to_file_map({"header": nib.FileHolder(fileobj=buf),
                     "image":  nib.FileHolder(fileobj=buf)})
    return buf.getvalue()

def mask_preview_base64(mask_vol: np.ndarray) -> str:
    """Middle axial slice of binary mask → base64 PNG."""
    sl = get_middle_slice(mask_vol)
    rgba = np.zeros((*sl.shape, 4), dtype=np.uint8)
    rgba[sl > 0.5] = [255, 0, 0, 180]   # red overlay for tumor
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────
# PREPROCESSING FOR EACH MODEL
# ─────────────────────────────────────────────
clf_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def prepare_clf_input(flair: np.ndarray, t1ce: np.ndarray) -> torch.Tensor:
    """
    Stack middle slices of FLAIR + T1CE into a 3-channel RGB-like image
    for the EfficientNet classifier.
    """
    flair_n = normalize_volume(flair)
    t1ce_n  = normalize_volume(t1ce)

    flair_sl = get_middle_slice(flair_n)   # H x W
    t1ce_sl  = get_middle_slice(t1ce_n)    # H x W
    avg_sl   = (flair_sl + t1ce_sl) / 2.0  # average as 3rd channel

    rgb = np.stack([flair_sl, t1ce_sl, avg_sl], axis=-1)  # H x W x 3
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    pil_img = Image.fromarray(rgb, mode="RGB")
    tensor  = clf_transform(pil_img).unsqueeze(0).to(DEVICE)  # 1 x 3 x 224 x 224
    return tensor

def prepare_seg_input(flair: np.ndarray, t1ce: np.ndarray,
                      target_shape=(128, 128, 128)) -> np.ndarray:
    """
    Resize FLAIR + T1CE volumes to a fixed shape, stack as 2-channel input.
    Returns array of shape (1, D, H, W, 2) for the Keras model.
    Adjust channel/shape ordering below to match how YOUR model was trained.
    """
    from scipy.ndimage import zoom

    def resize_vol(vol, target):
        factors = [t / s for t, s in zip(target, vol.shape)]
        return zoom(vol, factors, order=1)

    flair_r = resize_vol(normalize_volume(flair), target_shape)
    t1ce_r  = resize_vol(normalize_volume(t1ce),  target_shape)

    combined = np.stack([flair_r, t1ce_r], axis=-1)  # D x H x W x 2
    return combined[np.newaxis, ...]                  # 1 x D x H x W x 2


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def run_classification(flair: np.ndarray, t1ce: np.ndarray) -> dict:
    tensor = prepare_clf_input(flair, t1ce)
    with torch.no_grad():
        logits = clf_model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence":      round(float(probs[idx]), 4),
        "probabilities": {
            "glioma":     round(float(probs[0]), 4),
            "meningioma": round(float(probs[1]), 4),
        }
    }

def run_segmentation(flair: np.ndarray, t1ce: np.ndarray) -> dict:
    inp        = prepare_seg_input(flair, t1ce)           # 1 x D x H x W x 2
    pred       = seg_model.predict(inp, verbose=0)        # 1 x D x H x W x 1  (or similar)
    mask_vol   = (pred[0, ..., 0] > 0.5).astype(np.float32)  # D x H x W binary

    # Stats
    voxel_count = int(mask_vol.sum())
    total_vox   = int(mask_vol.size)

    # NIfTI bytes (base64 so it travels over JSON)
    nii_bytes   = volume_to_nifti_bytes(mask_vol)
    nii_b64     = base64.b64encode(nii_bytes).decode("utf-8")

    # 2-D preview PNG (base64)
    preview_b64 = mask_preview_base64(mask_vol)

    return {
        "tumor_voxel_count":  voxel_count,
        "total_voxels":       total_vox,
        "tumor_volume_pct":   round(100.0 * voxel_count / total_vox, 4),
        "mask_nifti_b64":     nii_b64,       # decode & save as .nii.gz client-side
        "mask_preview_png_b64": preview_b64, # data:image/png;base64,<this>
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service":   "Brain Tumor Classification & Segmentation API",
        "models": {
            "classification": "classification.pth  (PyTorch EfficientNet-B0)",
            "segmentation":   "finetuned_meningioma_model.h5  (TF/Keras 3-D U-Net)"
        },
        "endpoints": {
            "POST /predict":             "Classification + Segmentation (combined)",
            "POST /predict/classify":    "Classification only",
            "POST /predict/segment":     "Segmentation only",
            "GET  /health":              "Health check"
        },
        "input": "Multipart form-data with fields: 'flair' (.nii or .nii.gz) and 't1ce' (.nii or .nii.gz)"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":              "ok",
        "classification_ready": clf_model is not None,
        "segmentation_ready":   seg_model is not None,
    })


def get_nifti_inputs():
    """Parse flair + t1ce from request files. Raises ValueError on missing."""
    if "flair" not in request.files or "t1ce" not in request.files:
        raise ValueError("Both 'flair' and 't1ce' NIfTI files are required.")
    flair = load_nifti_volume(request.files["flair"])
    t1ce  = load_nifti_volume(request.files["t1ce"])
    return flair, t1ce


@app.route("/predict", methods=["POST"])
def predict_both():
    """Run classification AND segmentation, return combined JSON."""
    try:
        flair, t1ce = get_nifti_inputs()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = {}

    if clf_model:
        try:
            result["classification"] = run_classification(flair, t1ce)
        except Exception as e:
            result["classification"] = {"error": str(e)}
    else:
        result["classification"] = {"error": "Classification model not loaded"}

    if seg_model:
        try:
            result["segmentation"] = run_segmentation(flair, t1ce)
        except Exception as e:
            result["segmentation"] = {"error": str(e)}
    else:
        result["segmentation"] = {"error": "Segmentation model not loaded"}

    return jsonify(result)


@app.route("/predict/classify", methods=["POST"])
def predict_classify():
    """Classification only."""
    if not clf_model:
        return jsonify({"error": "Classification model not loaded"}), 503
    try:
        flair, t1ce = get_nifti_inputs()
        return jsonify(run_classification(flair, t1ce))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/segment", methods=["POST"])
def predict_segment():
    """Segmentation only."""
    if not seg_model:
        return jsonify({"error": "Segmentation model not loaded"}), 503
    try:
        flair, t1ce = get_nifti_inputs()
        return jsonify(run_segmentation(flair, t1ce))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
