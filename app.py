import os
import io
import base64
import tempfile
import numpy as np
from flask import Flask, request, jsonify, render_template_string

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

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PTH_PATH    = os.path.join(BASE_DIR, "classification.pth")
H5_PATH     = os.path.join(BASE_DIR, "finetuned_meningioma_model.h5")


# ─────────────────────────────────────────────
# GLOBAL JSON ERROR HANDLERS
# Ensures the frontend NEVER receives an HTML error page.
# Without these, any unhandled exception returns Flask's default
# HTML 500 page, which causes "Unexpected token '<'" in the browser.
# ─────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Catch-all: any unhandled Python exception → JSON."""
    import traceback
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
def load_pytorch_model():
    if not os.path.exists(PTH_PATH):
        raise FileNotFoundError(
            f"classification.pth not found at {PTH_PATH}\n"
            f"Files in BASE_DIR: {os.listdir(BASE_DIR)}"
        )
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    state = torch.load(PTH_PATH, map_location=DEVICE)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def load_keras_model():
    if not os.path.exists(H5_PATH):
        raise FileNotFoundError(
            f"finetuned_meningioma_model.h5 not found at {H5_PATH}\n"
            f"Files in BASE_DIR: {os.listdir(BASE_DIR)}"
        )
    return tf.keras.models.load_model(H5_PATH)


print("=" * 60)
print(f"BASE_DIR       : {BASE_DIR}")
print(f"PTH_PATH       : {PTH_PATH}  exists={os.path.exists(PTH_PATH)}")
print(f"H5_PATH        : {H5_PATH}  exists={os.path.exists(H5_PATH)}")
print(f"Files in dir   : {os.listdir(BASE_DIR)}")
print(f"DEVICE         : {DEVICE}")
print("=" * 60)

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
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name
    vol = nib.load(tmp_path).get_fdata()
    os.unlink(tmp_path)
    return vol.astype(np.float32)

def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return (vol - vmin) / (vmax - vmin)

def get_middle_slice(vol: np.ndarray) -> np.ndarray:
    z = vol.shape[2] // 2
    return vol[:, :, z]

def slice_to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    s = np.clip(slice_2d, 0, 1)
    uint8 = (s * 255).astype(np.uint8)
    return np.stack([uint8, uint8, uint8], axis=-1)

def volume_to_nifti_bytes(vol: np.ndarray) -> bytes:
    nii = nib.Nifti1Image(vol, affine=np.eye(4))
    buf = io.BytesIO()
    nii.to_file_map({"header": nib.FileHolder(fileobj=buf),
                     "image":  nib.FileHolder(fileobj=buf)})
    return buf.getvalue()

def mask_preview_base64(mask_vol: np.ndarray) -> str:
    sl = get_middle_slice(mask_vol)
    rgba = np.zeros((*sl.shape, 4), dtype=np.uint8)
    rgba[sl > 0.5] = [255, 0, 0, 180]
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def slice_preview_base64(vol: np.ndarray) -> str:
    norm = normalize_volume(vol)
    sl   = get_middle_slice(norm)
    rgb  = slice_to_rgb(sl)
    img  = Image.fromarray(rgb, mode="RGB")
    img  = img.resize((256, 256), Image.LANCZOS)
    buf  = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
clf_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def prepare_clf_input(flair: np.ndarray, t1ce: np.ndarray) -> torch.Tensor:
    flair_n  = normalize_volume(flair)
    t1ce_n   = normalize_volume(t1ce)
    flair_sl = get_middle_slice(flair_n)
    t1ce_sl  = get_middle_slice(t1ce_n)
    avg_sl   = (flair_sl + t1ce_sl) / 2.0
    rgb      = np.stack([flair_sl, t1ce_sl, avg_sl], axis=-1)
    rgb      = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    pil_img  = Image.fromarray(rgb, mode="RGB")
    tensor   = clf_transform(pil_img).unsqueeze(0).to(DEVICE)
    return tensor

def prepare_seg_input(flair: np.ndarray, t1ce: np.ndarray,
                      target_shape=(128, 128, 128)) -> np.ndarray:
    from scipy.ndimage import zoom
    def resize_vol(vol, target):
        factors = [t / s for t, s in zip(target, vol.shape)]
        return zoom(vol, factors, order=1)
    flair_r  = resize_vol(normalize_volume(flair), target_shape)
    t1ce_r   = resize_vol(normalize_volume(t1ce),  target_shape)
    combined = np.stack([flair_r, t1ce_r], axis=-1)
    return combined[np.newaxis, ...]


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
    inp         = prepare_seg_input(flair, t1ce)
    pred        = seg_model.predict(inp, verbose=0)
    mask_vol    = (pred[0, ..., 0] > 0.5).astype(np.float32)
    voxel_count = int(mask_vol.sum())
    total_vox   = int(mask_vol.size)
    nii_bytes   = volume_to_nifti_bytes(mask_vol)
    nii_b64     = base64.b64encode(nii_bytes).decode("utf-8")
    preview_b64 = mask_preview_base64(mask_vol)
    return {
        "tumor_voxel_count":    voxel_count,
        "total_voxels":         total_vox,
        "tumor_volume_pct":     round(100.0 * voxel_count / total_vox, 4),
        "mask_nifti_b64":       nii_b64,
        "mask_preview_png_b64": preview_b64,
    }


# ─────────────────────────────────────────────
# HTML UI
# ─────────────────────────────────────────────
UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Brain Tumor Analysis</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --border: rgba(255,255,255,0.08);
    --border2: rgba(255,255,255,0.15);
    --text: #e8eaf6;
    --muted: #8b90a8;
    --accent: #7c6af7;
    --accent2: #9d8ff8;
    --success: #34d399;
    --warn: #fbbf24;
    --danger: #f87171;
    --glioma-color: #f87171;
    --menin-color: #818cf8;
    --radius: 12px;
    --radius-sm: 8px;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 2rem 1rem;
    line-height: 1.6;
  }

  .container { max-width: 880px; margin: 0 auto; }

  .header { text-align: center; margin-bottom: 2.5rem; }
  .header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
    background: linear-gradient(135deg, #a78bfa, #7c6af7, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .header p { color: var(--muted); font-size: 0.9rem; }

  .model-status {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-bottom: 2rem;
    flex-wrap: wrap;
  }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid;
  }
  .pill.ok  { border-color: rgba(52,211,153,0.4); background: rgba(52,211,153,0.08); color: var(--success); }
  .pill.err { border-color: rgba(248,113,113,0.4); background: rgba(248,113,113,0.08); color: var(--danger); }
  .pill-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
  }
  .card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
  }

  .upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 540px) { .upload-grid { grid-template-columns: 1fr; } }

  .upload-zone {
    border: 1.5px dashed var(--border2);
    border-radius: var(--radius-sm);
    padding: 1.4rem 1rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
  }
  .upload-zone:hover { border-color: var(--accent); background: rgba(124,106,247,0.05); }
  .upload-zone.has-file { border-color: var(--success); border-style: solid; }
  .upload-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .upload-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
  .upload-label { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.2rem; }
  .upload-hint  { font-size: 0.75rem; color: var(--muted); }
  .upload-name  { font-size: 0.78rem; color: var(--success); margin-top: 0.4rem; font-weight: 500; }

  .mode-row { display: flex; gap: 8px; margin-bottom: 1rem; flex-wrap: wrap; }
  .mode-btn {
    padding: 6px 16px;
    border-radius: 999px;
    border: 1px solid var(--border2);
    background: transparent;
    color: var(--muted);
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }
  .mode-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }

  .btn-run {
    width: 100%;
    padding: 0.85rem;
    border-radius: var(--radius-sm);
    border: none;
    background: var(--accent);
    color: #fff;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .btn-run:hover { opacity: 0.88; }
  .btn-run:active { transform: scale(0.99); }
  .btn-run:disabled { opacity: 0.4; cursor: not-allowed; }

  .spinner {
    width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #results { display: none; }

  .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 580px) { .result-grid { grid-template-columns: 1fr; } }

  .clf-result {
    background: var(--surface2);
    border-radius: var(--radius-sm);
    padding: 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .predicted-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    margin-bottom: 0.2rem;
  }
  .predicted-class { font-size: 1.5rem; font-weight: 700; text-transform: capitalize; }
  .predicted-class.glioma     { color: var(--glioma-color); }
  .predicted-class.meningioma { color: var(--menin-color); }

  .confidence-bar-wrap { margin-top: 0.2rem; }
  .confidence-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .bar-track { height: 6px; background: var(--border); border-radius: 999px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 999px; background: var(--accent); transition: width 0.6s ease; }

  .prob-row  { display: flex; flex-direction: column; gap: 8px; }
  .prob-item { display: flex; flex-direction: column; gap: 4px; }
  .prob-item-label { display: flex; justify-content: space-between; font-size: 0.78rem; }
  .prob-item-label span:first-child { color: var(--muted); text-transform: capitalize; }
  .prob-item-label span:last-child  { font-weight: 600; }

  .seg-result { background: var(--surface2); border-radius: var(--radius-sm); padding: 1.2rem; }
  .seg-preview {
    width: 100%;
    border-radius: 6px;
    border: 1px solid var(--border);
    margin-bottom: 1rem;
    image-rendering: pixelated;
    display: block;
  }
  .seg-stats { display: flex; flex-direction: column; gap: 8px; }
  .stat-row  { display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; }
  .stat-row span:first-child { color: var(--muted); }
  .stat-row span:last-child  { font-weight: 600; }

  .download-btn {
    width: 100%;
    margin-top: 1rem;
    padding: 8px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border2);
    background: transparent;
    color: var(--text);
    font-size: 0.82rem;
    cursor: pointer;
    transition: background 0.15s;
  }
  .download-btn:hover { background: var(--surface); }

  .error-box {
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.25);
    border-radius: var(--radius-sm);
    padding: 1rem 1.2rem;
    color: var(--danger);
    font-size: 0.85rem;
    display: none;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

  .slice-preview-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 0.75rem; }
  .slice-preview-item { text-align: center; }
  .slice-preview-item img { width: 100%; border-radius: 6px; border: 1px solid var(--border); display: block; }
  .slice-preview-item p  { font-size: 0.72rem; color: var(--muted); margin-top: 4px; }
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>Brain Tumor Analysis</h1>
    <p>Upload NIfTI MRI volumes for AI-powered classification and segmentation</p>
  </div>

  <div class="model-status">
    <div class="pill {{ 'ok' if clf_ready else 'err' }}">
      <span class="pill-dot"></span>
      Classification model {{ '(ready)' if clf_ready else '(not loaded)' }}
    </div>
    <div class="pill {{ 'ok' if seg_ready else 'err' }}">
      <span class="pill-dot"></span>
      Segmentation model {{ '(ready)' if seg_ready else '(not loaded)' }}
    </div>
  </div>

  <div class="card">
    <div class="card-title">Input volumes</div>
    <div class="upload-grid">
      <div class="upload-zone" id="flair-zone">
        <input type="file" id="flair-input" accept=".nii,.nii.gz" onchange="onFileChange('flair')"/>
        <div class="upload-icon">🧠</div>
        <div class="upload-label">FLAIR</div>
        <div class="upload-hint">.nii or .nii.gz</div>
        <div class="upload-name" id="flair-name"></div>
      </div>
      <div class="upload-zone" id="t1ce-zone">
        <input type="file" id="t1ce-input" accept=".nii,.nii.gz" onchange="onFileChange('t1ce')"/>
        <div class="upload-icon">💡</div>
        <div class="upload-label">T1CE</div>
        <div class="upload-hint">.nii or .nii.gz</div>
        <div class="upload-name" id="t1ce-name"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Analysis mode</div>
    <div class="mode-row">
      <button class="mode-btn active" id="mode-both"    onclick="setMode('both')">Classification + Segmentation</button>
      <button class="mode-btn"        id="mode-classify" onclick="setMode('classify')">Classification only</button>
      <button class="mode-btn"        id="mode-segment"  onclick="setMode('segment')">Segmentation only</button>
    </div>
    <button class="btn-run" id="run-btn" onclick="runAnalysis()" disabled>
      <div class="spinner" id="spinner"></div>
      <span id="run-label">Analyse</span>
    </button>
  </div>

  <div class="error-box" id="error-box"></div>

  <div id="results">
    <div class="section-label">Results</div>
    <div id="input-previews" style="display:none;" class="card">
      <div class="card-title">Input slices (axial middle)</div>
      <div class="slice-preview-grid">
        <div class="slice-preview-item">
          <img id="flair-preview-img" src="" alt="FLAIR slice"/>
          <p>FLAIR</p>
        </div>
        <div class="slice-preview-item">
          <img id="t1ce-preview-img" src="" alt="T1CE slice"/>
          <p>T1CE</p>
        </div>
      </div>
    </div>
    <div class="result-grid" id="result-grid"></div>
  </div>

</div>

<script>
  let currentMode  = 'both';
  let maskNiftiB64 = null;

  function setMode(mode) {
    currentMode = mode;
    ['both','classify','segment'].forEach(m => {
      document.getElementById('mode-' + m).classList.toggle('active', m === mode);
    });
  }

  function onFileChange(field) {
    const input = document.getElementById(field + '-input');
    const zone  = document.getElementById(field + '-zone');
    const name  = document.getElementById(field + '-name');
    if (input.files.length > 0) {
      name.textContent = input.files[0].name;
      zone.classList.add('has-file');
    } else {
      name.textContent = '';
      zone.classList.remove('has-file');
    }
    checkReady();
  }

  function checkReady() {
    const flair = document.getElementById('flair-input').files.length > 0;
    const t1ce  = document.getElementById('t1ce-input').files.length > 0;
    document.getElementById('run-btn').disabled = !(flair && t1ce);
  }

  function setLoading(loading) {
    const btn     = document.getElementById('run-btn');
    const spinner = document.getElementById('spinner');
    const label   = document.getElementById('run-label');
    btn.disabled          = loading;
    spinner.style.display = loading ? 'block' : 'none';
    label.textContent     = loading ? 'Analysing…' : 'Analyse';
  }

  function showError(msg) {
    const box = document.getElementById('error-box');
    box.textContent   = msg;
    box.style.display = 'block';
  }
  function hideError() {
    document.getElementById('error-box').style.display = 'none';
  }

  function pct(v) { return (v * 100).toFixed(1) + '%'; }

  function buildClassificationCard(data) {
    const cls   = data.predicted_class;
    const conf  = data.confidence;
    const probs = data.probabilities;
    return `
      <div class="clf-result">
        <div>
          <div class="predicted-label">Predicted tumor type</div>
          <div class="predicted-class ${cls}">${cls}</div>
        </div>
        <div class="confidence-bar-wrap">
          <div class="confidence-label">
            <span>Confidence</span><span>${pct(conf)}</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${pct(conf)};"></div>
          </div>
        </div>
        <div class="prob-row">
          <div class="prob-item">
            <div class="prob-item-label">
              <span>Glioma</span><span style="color:var(--glioma-color)">${pct(probs.glioma)}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width:${pct(probs.glioma)};background:var(--glioma-color);"></div>
            </div>
          </div>
          <div class="prob-item">
            <div class="prob-item-label">
              <span>Meningioma</span><span style="color:var(--menin-color)">${pct(probs.meningioma)}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width:${pct(probs.meningioma)};background:var(--menin-color);"></div>
            </div>
          </div>
        </div>
      </div>`;
  }

  function buildSegmentationCard(data) {
    maskNiftiB64 = data.mask_nifti_b64;
    const preview = data.mask_preview_png_b64;
    return `
      <div class="seg-result">
        <img class="seg-preview" src="data:image/png;base64,${preview}" alt="Segmentation mask"/>
        <div class="seg-stats">
          <div class="stat-row">
            <span>Tumor voxels</span>
            <span>${data.tumor_voxel_count.toLocaleString()}</span>
          </div>
          <div class="stat-row">
            <span>Total voxels</span>
            <span>${data.total_voxels.toLocaleString()}</span>
          </div>
          <div class="stat-row">
            <span>Tumor volume</span>
            <span>${data.tumor_volume_pct}%</span>
          </div>
        </div>
        <button class="download-btn" onclick="downloadMask()">⬇ Download mask (.nii.gz)</button>
      </div>`;
  }

  function downloadMask() {
    if (!maskNiftiB64) return;
    const bytes = atob(maskNiftiB64);
    const arr   = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    const blob = new Blob([arr], { type: 'application/octet-stream' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'tumor_mask.nii.gz'; a.click();
    URL.revokeObjectURL(url);
  }

  async function runAnalysis() {
    hideError();
    setLoading(true);
    document.getElementById('results').style.display = 'none';
    document.getElementById('result-grid').innerHTML = '';
    maskNiftiB64 = null;

    const flair = document.getElementById('flair-input').files[0];
    const t1ce  = document.getElementById('t1ce-input').files[0];

    const endpointMap = {
      both:     '/predict',
      classify: '/predict/classify',
      segment:  '/predict/segment',
    };

    const fd = new FormData();
    fd.append('flair', flair);
    fd.append('t1ce',  t1ce);

    try {
      const resp = await fetch(endpointMap[currentMode], { method: 'POST', body: fd });

      // Guard: if response is not JSON (server returned HTML), show a clear message
      const contentType = resp.headers.get('content-type') || '';
      if (!contentType.includes('application/json')) {
        const text = await resp.text();
        showError('Server returned a non-JSON response (HTTP ' + resp.status + ').\\n\\nFirst 400 chars:\\n' + text.slice(0, 400));
        setLoading(false);
        return;
      }

      const data = await resp.json();

      if (!resp.ok) {
        showError(data.error || 'Server error: ' + resp.status);
        setLoading(false);
        return;
      }

      const grid = document.getElementById('result-grid');
      let html = '';

      if (currentMode === 'both') {
        if (data.classification && !data.classification.error)
          html += `<div>${buildClassificationCard(data.classification)}</div>`;
        else if (data.classification?.error)
          html += `<div style="color:var(--danger);font-size:.85rem;padding:1rem;">${data.classification.error}</div>`;

        if (data.segmentation && !data.segmentation.error)
          html += `<div>${buildSegmentationCard(data.segmentation)}</div>`;
        else if (data.segmentation?.error)
          html += `<div style="color:var(--danger);font-size:.85rem;padding:1rem;">${data.segmentation.error}</div>`;
      } else if (currentMode === 'classify') {
        if (!data.error) html += `<div>${buildClassificationCard(data)}</div>`;
        else showError(data.error);
      } else if (currentMode === 'segment') {
        if (!data.error) html += `<div>${buildSegmentationCard(data)}</div>`;
        else showError(data.error);
      }

      grid.innerHTML = html;
      document.getElementById('results').style.display = 'block';
      document.getElementById('input-previews').style.display = 'none';

    } catch (err) {
      showError('Network error: ' + err.message);
    }

    setLoading(false);
  }
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        UI_HTML,
        clf_ready=clf_model is not None,
        seg_ready=seg_model is not None,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":               "ok",
        "classification_ready": clf_model is not None,
        "segmentation_ready":   seg_model is not None,
        "clf_path_exists":      os.path.exists(PTH_PATH),
        "seg_path_exists":      os.path.exists(H5_PATH),
        "base_dir":             BASE_DIR,
        "files_in_dir":         os.listdir(BASE_DIR),
    })


def get_nifti_inputs():
    if "flair" not in request.files or "t1ce" not in request.files:
        raise ValueError("Both 'flair' and 't1ce' NIfTI files are required.")
    flair = load_nifti_volume(request.files["flair"])
    t1ce  = load_nifti_volume(request.files["t1ce"])
    return flair, t1ce


@app.route("/predict", methods=["POST"])
def predict_both():
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
        result["classification"] = {"error": "Classification model not loaded — check Render logs for details"}

    if seg_model:
        try:
            result["segmentation"] = run_segmentation(flair, t1ce)
        except Exception as e:
            result["segmentation"] = {"error": str(e)}
    else:
        result["segmentation"] = {"error": "Segmentation model not loaded — check Render logs for details"}

    return jsonify(result)


@app.route("/predict/classify", methods=["POST"])
def predict_classify():
    if not clf_model:
        return jsonify({"error": "Classification model not loaded — check Render logs for details"}), 503
    try:
        flair, t1ce = get_nifti_inputs()
        return jsonify(run_classification(flair, t1ce))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/segment", methods=["POST"])
def predict_segment():
    if not seg_model:
        return jsonify({"error": "Segmentation model not loaded — check Render logs for details"}), 503
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
