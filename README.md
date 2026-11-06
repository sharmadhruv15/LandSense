# 🏔️ Himalayan Landslide Detection

AI-powered landslide risk analysis using U-Net trained on Landslide4Sense (14-band satellite data).

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open: http://localhost:5000
```

## Model Setup (required)

Place these 3 files in `outputs/checkpoints/` after training:
- `best_unet.pth`     — trained model weights
- `band_mean.npy`     — per-band mean (saved by training notebook)
- `band_std.npy`      — per-band std  (saved by training notebook)

## Three Input Modes

| Mode | Input | Accuracy | Use Case |
|------|-------|----------|----------|
| **H5 Satellite** | `.h5` from Landslide4Sense | ⭐⭐⭐ Highest | Research / agency use |
| **RGB Image** | PNG/JPG satellite photo | ⭐⭐ Good | Field screening |
| **Draw on Map** | Rectangle on Leaflet map | ⭐⭐ Good | Quick area check |

## Model Performance

- **Best IoU:** 0.5778
- **Best Dice:** 0.7282
- **Training:** 40 epochs, Focal+Dice loss, CosineAnnealing LR, TTA inference
- **Architecture:** U-Net (14-channel input, 2-class output)

## Tech Stack

- **Backend:** Python, Flask, PyTorch, NumPy, SciPy, h5py, Pillow
- **Frontend:** HTML, CSS, Vanilla JS, Leaflet.js
- **Model:** U-Net with Dropout (Landslide4Sense dataset)

## Training

Open `notebooks/improved_training.ipynb` in Google Colab (T4 GPU recommended).
