# U-DRS Quick Start Guide

Get up and running with U-DRS in 5 minutes!

---

## Step 1: Install Dependencies (2 minutes)

```bash
cd U-DRS
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning)
- OpenCV (image processing)
- Open3D (3D reconstruction)
- FastAPI (web server)
- And more...

---

## Step 2: Generate Sample Images (30 seconds)

```bash
python create_samples.py
```

This creates test images in `data/samples/`:
- `crack_1.jpg` through `crack_5.jpg`
- `dent_1.jpg` through `dent_5.jpg`
- `corrosion_1.jpg` through `corrosion_5.jpg`
- `intact.jpg`

---

## Step 3: Run Your First Analysis (1 minute)

```bash
python run_inference.py --input data/samples/crack_1.jpg
```

**Output:**
```
============================================================
DAMAGE ANALYSIS RESULTS
============================================================

✓ Damage detected (confidence: 92.5%)

2D MEASUREMENTS:
  • Damage area: 1250.5 mm²
  • Crack length: 245.8 mm
  • Crack width (mean): 3.2 mm
  • Crack width (max): 5.1 mm
  • Bounding box: 120.5 × 85.3 mm

3D MEASUREMENTS:
  • Max depth: 15.3 mm
  • Mean depth: 8.2 mm
  • Volume: 2580.4 mm³
  • Max deformation: 14.2 mm

SEVERITY ASSESSMENT:
  • Class: SEVERE
  • Score: 62.5 / 100
  • Confidence: 85.0%

COST ESTIMATION:
  • Est. cost: $485.50
  • Range: $388.40 - $582.60

REPAIR URGENCY:
  • Level: URGENT
  • Timeline: 1-2 weeks
  • Repair needed within 1-2 weeks

PROCESSING TIME:
  • Detection: 0.05s
  • Segmentation: 0.08s
  • Depth: 0.15s
  • 3D Reconstruction: 0.30s
  • Total: 0.64s

============================================================
Results saved to: data/outputs/crack_1
  • Full report: data/outputs/crack_1/report.json
  • Mask: data/outputs/crack_1/mask.png
  • Overlay: data/outputs/crack_1/overlay.png
  • Depth map: data/outputs/crack_1/depth_map.png
  • Point cloud: data/outputs/crack_1/point_cloud.ply
  • Mesh: data/outputs/crack_1/mesh.ply
============================================================
```

---

## Step 4: View 3D Model (30 seconds)

Open the 3D model in any mesh viewer:
- **MeshLab**: Free, cross-platform
- **Blender**: Professional 3D software
- **Online**: [https://3dviewer.net/](https://3dviewer.net/)

```bash
# Using Open3D viewer (built-in)
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('data/outputs/crack_1/point_cloud.ply')])"
```

---

## Step 5: Start API Server (30 seconds)

```bash
uvicorn api.main:app --reload --port 8000
```

**Visit:**
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

**Test API:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@data/samples/crack_1.jpg" \
  -F "pixel_to_mm_ratio=0.5"
```

---

## Next Steps

### Calibration
Real-world measurements require calibration:

```bash
# If 1 pixel = 0.5mm
python run_inference.py --input crack.jpg --pixel-mm-ratio 0.5

# For depth calibration
python run_inference.py --input dent.jpg --depth-scale 2.0
```

### Batch Processing
Process multiple images:

```python
from pathlib import Path
from pipeline.inference import create_pipeline

pipeline = create_pipeline()

for img_path in Path("data/samples").glob("crack_*.jpg"):
    results = pipeline.process(img_path)
    print(f"{img_path.name}: {results['severity']['class']}")
```

### Generate More Data
Create a large synthetic dataset:

```bash
python utils/dataset_generator.py --num-samples 1000 --output data/synthetic
```

### Train Models
Train your own segmentation model:

```bash
python models/segmentation/segmentation_train.py \
  --data data/synthetic \
  --epochs 50 \
  --batch-size 16
```

---

## Common Options

### CLI Flags
```bash
# Skip 3D (faster)
python run_inference.py --input crack.jpg --no-3d

# Use CPU only
python run_inference.py --input crack.jpg --device cpu

# Verbose logging
python run_inference.py --input crack.jpg --verbose

# Custom output directory
python run_inference.py --input crack.jpg --output results/analysis_001
```

### Configuration
Edit `pipeline/config.py`:

```python
# Use smaller depth model for speed
DEPTH_MODEL_TYPE = "MiDaS_small"  # Default: "DPT_Large"

# Reduce resolution
INPUT_SIZE = (320, 240)  # Default: (640, 480)

# Adjust thresholds
DETECTION_THRESHOLD = 0.6  # Default: 0.5
SEGMENTATION_THRESHOLD = 0.6  # Default: 0.5
```

---

## Troubleshooting

### Out of Memory
```python
# In config.py
INPUT_SIZE = (320, 240)  # Smaller resolution
DEPTH_MODEL_TYPE = "MiDaS_small"  # Lighter model
```

### Slow Inference
```bash
# GPU check
nvidia-smi

# Use GPU if available
python run_inference.py --input crack.jpg --device cuda

# Skip 3D reconstruction
python run_inference.py --input crack.jpg --no-3d
```

### Models Not Downloading
```python
# Manual download (run once)
import torch
torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
```

---

## What's in the Output?

For each analysis, you get:

**JSON Report** (`report.json`):
- Complete measurements (2D + 3D)
- Severity score and classification
- Cost estimation
- Timing breakdown

**Images**:
- `mask.png` - Binary segmentation mask
- `overlay.png` - Damage highlighted on original
- `edges.png` - Edge detection result
- `depth_map.png` - Depth visualization

**3D Models**:
- `point_cloud.ply` - 3D point cloud
- `mesh.ply` - Reconstructed surface

---

## Ready for More?

- 📖 [Full Documentation](README.md)
- 🏗️ [Architecture Guide](docs/ARCHITECTURE.md)
- 🌐 [API Reference](docs/API_GUIDE.md)
- 🚀 [Deployment Guide](docs/DEPLOYMENT.md)

---

**Enjoy U-DRS! 🎉**
