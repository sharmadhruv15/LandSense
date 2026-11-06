import os
import sys
import torch
import time
import tempfile
import numpy as np
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, sobel
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from models.unet import UNet

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# ---------------------------------------------------------------------------
# Band statistics
# ---------------------------------------------------------------------------
_CKPT_DIR  = os.path.join(os.path.dirname(__file__), 'outputs', 'checkpoints')
_MEAN_PATH = os.path.join(_CKPT_DIR, 'band_mean.npy')
_STD_PATH  = os.path.join(_CKPT_DIR, 'band_std.npy')

_FALLBACK_MEAN = np.array([
    0.4353, 0.4078, 0.3922, 0.4196, 0.3725, 0.3098,
    0.4039, 0.3882, 0.4196, 0.4039, 0.3333, 0.2000,
    0.1765, 0.3137
], dtype=np.float32)

_FALLBACK_STD = np.array([
    0.1569, 0.1490, 0.1373, 0.1608, 0.1490, 0.1294,
    0.1529, 0.1451, 0.1569, 0.1529, 0.1255, 0.1216,
    0.1098, 0.1373
], dtype=np.float32)

if os.path.exists(_MEAN_PATH) and os.path.exists(_STD_PATH):
    L4S_MEAN = np.load(_MEAN_PATH).astype(np.float32)
    L4S_STD  = np.load(_STD_PATH).astype(np.float32)
    print(f"✅ Loaded band stats from training: {_MEAN_PATH}")
else:
    L4S_MEAN = _FALLBACK_MEAN
    L4S_STD  = _FALLBACK_STD
    print("⚠️  band_mean.npy not found — using fallback stats.")


# ---------------------------------------------------------------------------
# Utility: Test-Time Augmentation (TTA)
# Averages predictions over 8 augmentations (4 rotations x 2 flips).
# Adds ~+0.02 IoU for free with no retraining.
# ---------------------------------------------------------------------------
def predict_with_tta(model, img_tensor, device):
    probs_sum = None
    for flip in [False, True]:
        for k in range(4):
            x = img_tensor.clone()
            if flip:
                x = torch.flip(x, dims=[3])
            if k > 0:
                x = torch.rot90(x, k=k, dims=[2, 3])
            with torch.no_grad():
                prob = torch.softmax(model(x), dim=1)[0, 1].cpu()
            if k > 0:
                prob = torch.rot90(prob, k=-k, dims=[0, 1])
            if flip:
                prob = torch.flip(prob, dims=[1])
            probs_sum = prob if probs_sum is None else probs_sum + prob
    return (probs_sum / 8).numpy()   # (H, W) float in [0,1]


# ---------------------------------------------------------------------------
# Utility: RGB → 14-band tensor (for RGB image uploads)
# ---------------------------------------------------------------------------
def rgb_to_14band(img_rgb_array):
    """
    Convert (H,W,3) float32 [0,1] RGB to (1,14,H,W) tensor.
    Maps R->B3, G->B2, B->B1. Fills remaining 11 bands with
    dataset mean (neutral prior). Applies z-score normalisation.
    """
    H, W = img_rgb_array.shape[:2]
    tensor_14 = np.zeros((14, H, W), dtype=np.float32)
    tensor_14[0] = img_rgb_array[:, :, 2]   # B1 <- Blue
    tensor_14[1] = img_rgb_array[:, :, 1]   # B2 <- Green
    tensor_14[2] = img_rgb_array[:, :, 0]   # B3 <- Red
    for b in range(3, 14):
        tensor_14[b] = L4S_MEAN[b]          # neutral prior for missing bands
    for b in range(14):
        tensor_14[b] = (tensor_14[b] - L4S_MEAN[b]) / (L4S_STD[b] + 1e-8)
    return torch.from_numpy(tensor_14).unsqueeze(0)   # (1,14,H,W)


# ---------------------------------------------------------------------------
# Terrain type detector (used by RGB mode only)
# ---------------------------------------------------------------------------
def assess_terrain_type(img_rgb_uint8):
    gray = np.mean(img_rgb_uint8.astype(np.float32), axis=2)

    # Exclude road pixels (dark grey/black) before computing slope
    r = img_rgb_uint8[:, :, 0].astype(float)
    g = img_rgb_uint8[:, :, 1].astype(float)
    b = img_rgb_uint8[:, :, 2].astype(float)
    brightness = (r + g + b) / 3.0
    road_mask = (brightness < 45) & (np.abs(r - g) < 15) & (np.abs(g - b) < 15)

    gray_no_road = gray.copy()
    gray_no_road[road_mask] = np.nan

    dx = sobel(gray, axis=0)
    dy = sobel(gray, axis=1)
    grad_mag = np.hypot(dx, dy)

    slope_mean    = float(np.nanmean(grad_mag))
    slope_norm    = float(np.clip(slope_mean / 30.0, 0, 1))
    flatness_score = 1.0 - slope_norm

    green_pct      = float(np.mean((g > 80) & (g > r * 0.9) & (g > b * 0.9)))
    brightness_std = float(np.std(gray))
    is_farmland    = (green_pct > 0.30) and (slope_mean < 12) and (brightness_std < 55)
    is_flat        = (slope_mean < 10) or (flatness_score > 0.72) or is_farmland

    if is_flat and is_farmland:
        terrain_label = "Flat farmland / plain (near-zero landslide risk)"
    elif is_flat:
        terrain_label = "Flat terrain / valley floor"
    elif flatness_score > 0.45:
        terrain_label = "Gently sloping terrain"
    else:
        terrain_label = "Steep / mountainous terrain"

    return {
        'is_flat': is_flat,
        'flatness_score': round(flatness_score, 3),
        'terrain_label': terrain_label,
        'slope_mean': round(slope_mean, 2),
        'is_farmland': is_farmland
    }


def apply_terrain_cap(risk_score, fused_map, terrain_info):
    if not terrain_info['is_flat']:
        return risk_score, fused_map
    flat_cap     = 0.20
    fused_capped = np.clip(fused_map, 0, flat_cap)
    flatness     = terrain_info['flatness_score']
    max_score    = int((1.0 - flatness) * 25)
    return min(risk_score, max_score), fused_capped


# ---------------------------------------------------------------------------
# Heuristic risk map (RGB only) — improved bare-soil detection
# ---------------------------------------------------------------------------
def compute_heuristic_risk(img_rgb_uint8):
    img = img_rgb_uint8.astype(np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    ndvi_proxy = (g - r) / (g + r + 1e-6)
    brightness = (r + g + b) / 3.0

    # ── Improved bare soil: catches brown, tan, grey, sandy exposed earth ──
    bare_soil = (ndvi_proxy < 0.02) & (brightness > 35) & (brightness < 200)
    bare_rock  = bare_soil & ~((b > r + 15) & (b > g + 10))  # exclude water

    # Snow/ice: very bright, neutral colour
    snow  = (brightness > 180) & (np.abs(r - g) < 25) & (np.abs(g - b) < 25)

    # Water: blue dominant, dark
    water = (b > r + 20) & (b > g + 10) & (brightness < 160)

    # Road: dark grey/black (exclude from risk)
    road  = (brightness < 45) & (np.abs(r - g) < 15) & (np.abs(g - b) < 15)

    # Vegetation: clearly green
    vegetation = ndvi_proxy > 0.08

    # ── Build risk map ──
    risk = np.zeros(img.shape[:2], dtype=np.float32)

    # Base: inverted NDVI (how bare is each pixel)
    bare_intensity = np.clip((-ndvi_proxy + 0.1) / 0.4, 0, 1)
    risk += bare_intensity * 0.6

    # Bare rock/soil bonus
    risk[bare_rock] = np.clip(risk[bare_rock] + 0.3, 0, 1)

    # Erosion gully: dark narrow features surrounded by bare soil
    # (dangerous water channels that destabilise slopes)
    very_dark = (brightness < 40) & (np.abs(r - b) < 20) & ~road
    risk[very_dark] = np.clip(risk[very_dark] + 0.15, 0, 1)

    # Penalise safe areas
    risk[vegetation] = np.clip(risk[vegetation] - 0.4, 0, 1)
    risk[water]      = np.clip(risk[water]      - 0.5, 0, 1)
    risk[snow]       = np.clip(risk[snow]       - 0.15, 0, 1)
    risk[road]       = np.clip(risk[road]       - 0.3, 0, 1)

    risk = gaussian_filter(risk, sigma=3)
    return np.clip(risk, 0, 1)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def fuse_model_and_heuristic(model_prob, heuristic_prob, model_weight=0.65):
    return np.clip(model_weight * model_prob + (1 - model_weight) * heuristic_prob, 0, 1)


def prob_to_risk_category(score_0_100):
    if score_0_100 >= 65:
        return "VERY HIGH", "AVOID — Very high landslide probability. Do not travel during monsoon."
    elif score_0_100 >= 45:
        return "HIGH",      "CAUTION — High landslide risk. Monitor weather and avoid steep slopes."
    elif score_0_100 >= 25:
        return "MODERATE",  "BE ALERT — Moderate risk. Check weather forecast before travel."
    else:
        return "LOW",       "SAFE — Low landslide risk. Generally safe for normal activities."


def build_overlay_image(img_rgb_uint8, fused_prob_map):
    overlay  = img_rgb_uint8.astype(np.float32).copy()
    colours  = {
        'very_high': np.array([220,  50,  40], dtype=np.float32),
        'high':      np.array([230, 140,  40], dtype=np.float32),
        'moderate':  np.array([241, 196,  15], dtype=np.float32),
    }
    very_high  = fused_prob_map > 0.65
    high       = (fused_prob_map > 0.45) & ~very_high
    moderate   = (fused_prob_map > 0.25) & ~very_high & ~high
    safe       = fused_prob_map <= 0.25

    colour_layer = np.zeros_like(overlay)
    alpha_map    = np.zeros(fused_prob_map.shape, dtype=np.float32)
    colour_layer[very_high] = colours['very_high']
    colour_layer[high]      = colours['high']
    colour_layer[moderate]  = colours['moderate']
    alpha_map[~safe]        = 0.50

    alpha_3ch = np.stack([alpha_map] * 3, axis=-1)
    return (overlay * (1 - alpha_3ch) + colour_layer * alpha_3ch).clip(0, 255).astype(np.uint8)


def build_cat_map(prob_map):
    cat = np.zeros_like(prob_map)
    cat[(prob_map > 0.25) & (prob_map <= 0.45)] = 1
    cat[(prob_map > 0.45) & (prob_map <= 0.65)] = 2
    cat[prob_map > 0.65]                         = 3
    return cat


def generate_dashboard_image(img_display_array, prob_map, cat_map, extra_band=None, extra_label='NIR band'):
    """
    img_display_array : (H,W,3) uint8 RGB for display
    prob_map          : (H,W) float [0,1]
    cat_map           : (H,W) int   0-3
    extra_band        : optional (H,W) float — shown in panel 3 (NIR or SAR)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Landslide Risk Analysis Dashboard', fontsize=18)

    axes[0, 0].imshow(img_display_array.astype(np.uint8))
    axes[0, 0].set_title('RGB Image');  axes[0, 0].axis('off')

    gray = np.mean(img_display_array, axis=2)
    dx   = sobel(gray, axis=0);  dy = sobel(gray, axis=1)
    slope_vis = gaussian_filter(np.hypot(dx, dy), sigma=2)
    axes[0, 1].imshow(slope_vis, cmap='terrain')
    axes[0, 1].set_title('Slope proxy'); axes[0, 1].axis('off')

    if extra_band is not None:
        axes[0, 2].imshow(extra_band, cmap='RdYlGn')
        axes[0, 2].set_title(extra_label)
    else:
        heuristic = compute_heuristic_risk(img_display_array.astype(np.uint8))
        axes[0, 2].imshow(heuristic, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[0, 2].set_title('Heuristic Risk Map')
    axes[0, 2].axis('off')

    im = axes[1, 0].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Model Landslide Probability')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    cmap_cat = mcolors.ListedColormap(['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
    axes[1, 1].imshow(cat_map, cmap=cmap_cat, vmin=0, vmax=3)
    axes[1, 1].set_title('Final Risk Map\n(Green=Low → Red=Very High)')
    axes[1, 1].axis('off')

    axes[1, 2].axis('off')
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Low Risk  (≤25%)'),
        mpatches.Patch(color='#f1c40f', label='Moderate  (25-45%)'),
        mpatches.Patch(color='#e67e22', label='High      (45-65%)'),
        mpatches.Patch(color='#e74c3c', label='Very High (>65%)')
    ]
    axes[1, 2].legend(handles=legend_elements, loc='center',
                      title="Risk Legend", fontsize=12, title_fontsize=14)
    plt.tight_layout()

    timestamp      = int(time.time())
    local_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"risk_map_{timestamp}.png")
    plt.savefig(local_filename, bbox_inches='tight', dpi=100)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    dashboard_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return dashboard_b64


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model      = UNet(n_channels=14, n_classes=2).to(device)
model_path = 'outputs/checkpoints/best_unet.pth'
MODEL_LOADED = False

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    MODEL_LOADED = True
    print("✅ U-Net model loaded successfully!")
else:
    print(f"⚠️  Model not found at {model_path}. Running in heuristic-only mode.")


# ---------------------------------------------------------------------------
# Analysis function 1: H5 file (all 14 bands — most accurate)
# ---------------------------------------------------------------------------
def analyze_h5_file(h5_bytes):
    """
    Full 14-band inference on a real Landslide4Sense .h5 file.
    This is the most accurate mode — NIR, SAR-VV, SAR-VH, Slope all used.
    """
    try:
        import h5py
    except ImportError:
        return {'error': 'h5py not installed. Run: pip install h5py'}

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp.write(h5_bytes)
        tmp_path = tmp.name

    try:
        with h5py.File(tmp_path, 'r') as f:
            img = np.array(f['img']).astype(np.float32)   # (128,128,14)

        H, W = img.shape[:2]

        # Z-score normalise all 14 bands
        img_norm   = (img - L4S_MEAN) / (L4S_STD + 1e-8)   # (128,128,14)
        img_tensor = torch.from_numpy(
            img_norm.transpose(2, 0, 1)
        ).unsqueeze(0).to(device)                            # (1,14,128,128)

        # TTA inference
        prob_map   = predict_with_tta(model, img_tensor, device)  # (128,128) [0,1]
        prob_smooth = gaussian_filter(prob_map, sigma=1.5)

        mean_prob = float(prob_smooth.mean())
        high_pct  = float(np.mean(prob_smooth > 0.65) * 100)
        mod_pct   = float(np.mean((prob_smooth > 0.45) & (prob_smooth <= 0.65)) * 100)

        risk_score = int(np.clip(mean_prob * 70 + high_pct * 0.30, 0, 100))
        risk_level, recommendation = prob_to_risk_category(risk_score)

        # Build RGB for display from bands 3,2,1 (de-normalised)
        rgb = img[:, :, [2, 1, 0]].copy()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # NIR band for dashboard
        nir_band = img[:, :, 3]   # band 4 (NIR) raw
        nir_disp = (nir_band - nir_band.min()) / (nir_band.max() - nir_band.min() + 1e-8)

        overlay     = build_overlay_image(rgb_uint8, prob_smooth)
        overlay_pil = Image.fromarray(overlay)
        cat_map     = build_cat_map(prob_smooth)

        dashboard_b64 = generate_dashboard_image(
            rgb_uint8, prob_smooth, cat_map,
            extra_band=nir_disp, extra_label='NIR Band (band 4)')

        buf = io.BytesIO()
        overlay_pil.save(buf, format='PNG')
        risk_map_b64 = base64.b64encode(buf.getvalue()).decode()

        # Band stats
        sar_vv_mean  = float(img[:, :, 11].mean())
        sar_vh_mean  = float(img[:, :, 12].mean())
        slope_val    = float(img[:, :, 13].mean())
        nir_mean     = float(img[:, :,  3].mean())

        return {
            'place':          'Analyzed Region (Full 14-band H5 — Most Accurate)',
            'risk_level':     risk_level,
            'risk_score':     risk_score,
            'factors':        (f'U-Net (all 14 bands + TTA): {mean_prob*100:.1f}% avg prob | '
                               f'{high_pct:.1f}% very-high pixels | {mod_pct:.1f}% high pixels'),
            'recommendation': recommendation,
            'risk_map':       risk_map_b64,
            'dashboard_map':  dashboard_b64,
            'image_analysis': {
                'dimensions':           f'{W}x{H}px',
                'bands_used':           '14/14 (RGB + NIR + SWIR + SAR-VV + SAR-VH + Slope)',
                'nir_mean':             round(nir_mean,    4),
                'sar_vv_mean':          round(sar_vv_mean, 4),
                'sar_vh_mean':          round(sar_vh_mean, 4),
                'slope_mean':           round(slope_val,   4),
                'landslide_probability': round(mean_prob * 100, 2),
                'model':                'U-Net 14-band + TTA (most accurate)'
            }
        }
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Analysis function 2: RGB image + model + heuristic fusion
# ---------------------------------------------------------------------------
def analyze_image_with_model(image_bytes):
    """RGB image: U-Net (3-band proxy) + heuristic fusion."""
    try:
        img_pil = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = img_pil.size
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        img_128       = img_pil.resize((128, 128), Image.BILINEAR)
        img_array_128 = np.array(img_128).astype(np.float32) / 255.0
        img_tensor    = rgb_to_14band(img_array_128).to(device)

        # TTA inference
        prob_map_128  = predict_with_tta(model, img_tensor, device)  # (128,128)
        prob_smooth   = gaussian_filter(prob_map_128, sigma=2.0)
        prob_full     = np.array(
            Image.fromarray((prob_smooth * 255).astype(np.uint8))
                 .resize((original_width, original_height), Image.BILINEAR)
        ).astype(np.float32) / 255.0

        mean_model_prob = float(prob_map_128.mean())
        print(f"  Model → mean={mean_model_prob*100:.2f}%  max={prob_map_128.max()*100:.2f}%")

        img_orig_array = np.array(img_pil).astype(np.uint8)
        heuristic_map  = compute_heuristic_risk(img_orig_array)
        fused          = fuse_model_and_heuristic(prob_full, heuristic_map, model_weight=0.65)

        terrain_info = assess_terrain_type(img_orig_array)
        print(f"  Terrain -> {terrain_info['terrain_label']}  slope_mean={terrain_info['slope_mean']}")

        high_pct   = float(np.mean(fused > 0.65) * 100)
        mod_pct    = float(np.mean((fused > 0.45) & (fused <= 0.65)) * 100)
        mean_fused = float(fused.mean())
        risk_score = int(np.clip(mean_fused * 70 + high_pct * 0.30, 0, 100))

        risk_score, fused = apply_terrain_cap(risk_score, fused, terrain_info)
        high_pct = float(np.mean(fused > 0.65) * 100)
        mod_pct  = float(np.mean((fused > 0.45) & (fused <= 0.65)) * 100)

        risk_level, recommendation = prob_to_risk_category(risk_score)
        if terrain_info['is_flat']:
            recommendation += (f" | Terrain: {terrain_info['terrain_label']} — "
                               "landslides require slope; this flat area has negligible risk.")

        cat_map     = build_cat_map(fused)
        overlay_img = build_overlay_image(img_orig_array, fused)
        overlay_pil = Image.fromarray(overlay_img)

        r = img_orig_array[:, :, 0].astype(float)
        g = img_orig_array[:, :, 1].astype(float)
        b = img_orig_array[:, :, 2].astype(float)
        vegetation_pct = float(round(np.mean((g > 100) & (g > r) & (g > b)) * 100, 1))
        bare_rock_pct  = float(round(np.mean((r > 100) & (g > 80) & (b < 120)) * 100, 1))
        water_pct      = float(round(np.mean((b > r + 20) & (b > g + 10)) * 100, 1))
        brightness     = float(round(np.mean(img_orig_array), 1))

        dashboard_b64 = generate_dashboard_image(
            np.array(img_pil).astype(np.float32), fused, cat_map)

        buf = io.BytesIO()
        overlay_pil.save(buf, format='PNG')
        risk_map_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            'place':          'Analyzed Region (AI + Heuristic Fusion)',
            'risk_level':     risk_level,
            'risk_score':     risk_score,
            'factors':        (f'U-Net: {mean_model_prob*100:.1f}% avg landslide prob | '
                               f'Fused: {mean_fused*100:.1f}% mean risk | '
                               f'{high_pct:.1f}% very-high-risk pixels, {mod_pct:.1f}% high-risk pixels'),
            'recommendation': recommendation,
            'risk_map':       risk_map_b64,
            'dashboard_map':  dashboard_b64,
            'image_analysis': {
                'dimensions':            f"{original_width}x{original_height}px",
                'aspect_ratio':          float(round(original_width / original_height, 2)),
                'vegetation_pct':        vegetation_pct,
                'bare_rock_pct':         bare_rock_pct,
                'water_pct':             water_pct,
                'brightness':            brightness,
                'model':                 'U-Net RGB proxy + Heuristic Fusion + TTA',
                'landslide_probability': round(mean_model_prob * 100, 2),
                'terrain':               terrain_info['terrain_label'],
                'flatness_score':        terrain_info['flatness_score']
            }
        }

    except Exception as e:
        import traceback
        print(f"Error in model prediction: {e}")
        traceback.print_exc()
        return analyze_image_heuristic(image_bytes)


# ---------------------------------------------------------------------------
# Analysis function 3: Heuristic only (no model)
# ---------------------------------------------------------------------------
def analyze_image_heuristic(image_bytes):
    img_pil = Image.open(io.BytesIO(image_bytes))
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    img_array = np.array(img_pil).astype(np.uint8)
    H, W = img_array.shape[:2]
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)

    vegetation_pct = float(round(np.mean((g > 100) & (g > r) & (g > b)) * 100, 1))
    bare_rock_pct  = float(round(np.mean((r > 100) & (g > 80) & (b < 120)) * 100, 1))
    water_pct      = float(round(np.mean((b > r + 20) & (b > g + 10)) * 100, 1))
    brightness     = float(round(np.mean(img_array), 1))

    heuristic_map = compute_heuristic_risk(img_array)
    terrain_info  = assess_terrain_type(img_array)
    print(f"  Terrain -> {terrain_info['terrain_label']}  slope_mean={terrain_info['slope_mean']}")

    mean_risk  = float(heuristic_map.mean())
    high_pct   = float(np.mean(heuristic_map > 0.65) * 100)
    risk_score = int(np.clip(mean_risk * 70 + high_pct * 0.30, 0, 100))

    risk_score, heuristic_map = apply_terrain_cap(risk_score, heuristic_map, terrain_info)
    risk_level, recommendation = prob_to_risk_category(risk_score)
    if terrain_info['is_flat']:
        recommendation += (f" | Terrain: {terrain_info['terrain_label']} — "
                           "landslides require slope; this flat area has negligible risk.")

    cat_map     = build_cat_map(heuristic_map)
    overlay_img = build_overlay_image(img_array, heuristic_map)
    overlay_pil = Image.fromarray(overlay_img)

    dashboard_b64 = generate_dashboard_image(
        img_array.astype(np.float32), heuristic_map, cat_map)

    buf = io.BytesIO()
    overlay_pil.save(buf, format='PNG')
    risk_map_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        'place':          'Analyzed Region (Heuristic Mode)',
        'risk_level':     risk_level,
        'risk_score':     risk_score,
        'factors':        f'Vegetation: {vegetation_pct}%, Bare Rock: {bare_rock_pct}%, Water: {water_pct}%',
        'recommendation': recommendation,
        'risk_map':       risk_map_b64,
        'dashboard_map':  dashboard_b64,
        'image_analysis': {
            'dimensions':     f"{W}x{H}px",
            'aspect_ratio':   round(W / H, 2),
            'vegetation_pct': vegetation_pct,
            'bare_rock_pct':  bare_rock_pct,
            'water_pct':      water_pct,
            'brightness':     brightness,
            'model':          'Heuristic only (model not loaded)',
            'terrain':        terrain_info['terrain_label'],
            'flatness_score': terrain_info['flatness_score']
        }
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file       = request.files['image']
    file_bytes = file.read()
    filename   = file.filename.lower()

    # Route based on file type
    if filename.endswith('.h5'):
        if not MODEL_LOADED:
            return jsonify({'error': 'Model not loaded — cannot process H5 files. '
                                     'Place best_unet.pth in outputs/checkpoints/'})
        print("H5 file detected — using full 14-band inference...")
        result = analyze_h5_file(file_bytes)

    else:
        # Standard RGB image (PNG, JPG, JPEG, TIFF, etc.)
        if MODEL_LOADED:
            print("RGB image — using U-Net + heuristic fusion...")
            result = analyze_image_with_model(file_bytes)
        else:
            print("Model not loaded — using heuristic only...")
            result = analyze_image_heuristic(file_bytes)

    return jsonify(result)


if __name__ == '__main__':
    print("=" * 50)
    print("🚀 LANDSLIDE RISK ANALYSIS SERVER")
    print("=" * 50)
    print(f"📍 Device      : {device}")
    print(f"📍 Model loaded: {MODEL_LOADED}")
    print("📍 Supported   : PNG/JPG (RGB) and .h5 (14-band)")
    print("📍 Open        : http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
