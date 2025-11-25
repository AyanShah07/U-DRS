# U-DRS API Guide

Complete reference for the U-DRS REST API.

---

## Base URL

```
http://localhost:8000
```

---

## Authentication

Currently, the API does not require authentication. For production deployment, consider adding:
- API keys
- OAuth 2.0
- JWT tokens

---

## Endpoints

### 1. Health Check

**GET** `/api/health`

Check if the API server is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cuda",
  "models_loaded": true
}
```

---

### 2. Analyze Damage

**POST** `/api/analyze`

Upload an image and perform complete damage analysis.

**Request:**
- Content-Type: `multipart/form-data`

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `file` | File | ✅ | - | Image file (JPG/PNG/BMP) |
| `pixel_to_mm_ratio` | float | ❌ | 1.0 | Calibration: pixels to mm |
| `depth_scale` | float | ❌ | 1.0 | Depth scale factor |
| `generate_3d` | boolean | ❌ | true | Generate 3D models |

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@crack.jpg" \
  -F "pixel_to_mm_ratio=0.5" \
  -F "depth_scale=1.5" \
  -F "generate_3d=true"
```

**Example (Python):**
```python
import requests

files = {"file": open("crack.jpg", "rb")}
data = {
    "pixel_to_mm_ratio": 0.5,
    "depth_scale": 1.5,
    "generate_3d": True
}

response = requests.post(
    "http://localhost:8000/api/analyze",
    files=files,
    data=data
)

result = response.json()
print(f"Analysis ID: {result['analysis_id']}")
print(f"Severity: {result['severity']['class']}")
```

**Response:**
```json
{
  "status": "damage_detected",
  "analysis_id": "a1b2c3d4-e5f6-...",
  "timestamp": "2024-01-15T10:30:00",
  "input_image": "/path/to/image.jpg",
  "detection": {
    "is_damaged": true,
    "confidence": 0.95,
    "probabilities": [0.05, 0.95]
  },
  "measurements": {
    "2d_measurements": {
      "area_mm2": 1250.5,
      "perimeter_mm": 180.2,
      "crack_length_mm": 245.8,
      "crack_width": {
        "mean": 3.2,
        "max": 5.1,
        "min": 1.8,
        "std": 0.9
      },
      "bounding_box": {
        "width": 120.5,
        "height": 85.3,
        "area": 10278.65,
        "aspect_ratio": 1.41
      },
      "compactness": 0.65,
      "circularity": 0.45,
      "num_regions": 1
    },
    "3d_measurements": {
      "depth_stats": {
        "max_mm": 15.3,
        "mean_mm": 8.2,
        "std_mm": 3.1,
        "range_mm": 14.5
      },
      "volume_from_depth_mm3": 2580.4,
      "deformation": {
        "deformation_volume_mm3": 2450.2,
        "mean_deformation_mm": 7.5,
        "max_deformation_mm": 14.2
      },
      "surface_area_mm2": 1320.6
    },
    "summary": {
      "damage_area_mm2": 1250.5,
      "crack_length_mm": 245.8,
      "crack_width_mean_mm": 3.2,
      "crack_width_max_mm": 5.1,
      "bbox_width_mm": 120.5,
      "bbox_height_mm": 85.3,
      "max_depth_mm": 15.3,
      "mean_depth_mm": 8.2,
      "volume_mm3": 2580.4,
      "max_deformation_mm": 14.2
    }
  },
  "severity": {
    "score": 62.5,
    "class": "severe",
    "confidence": 0.85,
    "threshold_low": 50.0,
    "threshold_high": 75.0
  },
  "cost_urgency": {
    "cost_prediction": {
      "estimated_cost_usd": 485.50,
      "lower_bound_usd": 388.40,
      "upper_bound_usd": 582.60
    },
    "urgency": "urgent",
    "urgency_description": "Repair needed within 1-2 weeks",
    "recommended_timeline": "1-2 weeks"
  },
  "timing": {
    "detection": 0.05,
    "segmentation": 0.08,
    "postprocessing": 0.02,
    "depth_estimation": 0.15,
    "3d_reconstruction": 0.30,
    "measurements": 0.02,
    "severity_scoring": 0.01,
    "cost_prediction": 0.01,
    "total": 0.64
  },
  "output_dir": "/path/to/outputs/analysis_id"
}
```

---

### 3. Get Analysis Result

**GET** `/api/result/{analysis_id}`

Retrieve completed analysis by ID.

**Parameters:**
- `analysis_id` (path): Unique analysis identifier

**Example:**
```bash
curl "http://localhost:8000/api/result/a1b2c3d4-e5f6-..."
```

**Response:** Same as `/api/analyze`

---

### 4. Download File

**GET** `/api/download/{analysis_id}/{file_type}`

Download specific output file from analysis.

**Parameters:**
- `analysis_id` (path): Analysis identifier
- `file_type` (path): One of:
  - `mask` - Binary segmentation mask
  - `depth_map` - Depth visualization
  - `overlay` - Damage overlay on original image
  - `edges` - Edge detection result
  - `point_cloud` - 3D point cloud (.ply)
  - `mesh` - 3D mesh (.ply)
  - `report` - JSON report

**Example:**
```bash
# Download mesh
curl "http://localhost:8000/api/download/a1b2c3d4.../mesh" \
  -o mesh.ply

# Download report
curl "http://localhost:8000/api/download/a1b2c3d4.../report" \
  -o report.json
```

**Response:** File download (binary)

---

### 5. List Analyses

**GET** `/api/list`

List all available analyses with metadata.

**Example:**
```bash
curl "http://localhost:8000/api/list"
```

**Response:**
```json
{
  "analyses": [
    {
      "analysis_id": "a1b2c3d4-...",
      "timestamp": "2024-01-15T10:30:00",
      "status": "damage_detected",
      "severity": "severe",
      "score": 62.5
    },
    {
      "analysis_id": "b2c3d4e5-...",
      "timestamp": "2024-01-15T09:15:00",
      "status": "no_damage",
      "severity": null,
      "score": null
    }
  ],
  "count": 2
}
```

---

### 6. Delete Analysis

**DELETE** `/api/result/{analysis_id}`

Delete analysis and all associated files.

**Parameters:**
- `analysis_id` (path): Analysis to delete

**Example:**
```bash
curl -X DELETE "http://localhost:8000/api/delete/a1b2c3d4-..."
```

**Response:**
```json
{
  "message": "Analysis a1b2c3d4-... deleted successfully"
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error description"
}
```

**Status Codes:**
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (analysis doesn't exist)
- `500` - Internal Server Error
- `503` - Service Unavailable (models not loaded)

---

## Rate Limiting

Currently no rate limiting is applied. For production:
- Implement per-IP rate limits
- Use API keys for tracking
- Consider queue-based processing for heavy loads

---

## WebSocket Support (Future)

For real-time progress updates during processing:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analyze');
ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(`Stage: ${progress.stage}, ${progress.percent}%`);
};
```

---

## Interactive Documentation

Visit `http://localhost:8000/docs` for:
- Interactive API explorer (Swagger UI)
- Try endpoints directly in browser
- Auto-generated request/response examples

---

## Client Libraries

### Python
```python
from udrs_client import UDRSClient

client = UDRSClient("http://localhost:8000")
result = client.analyze("crack.jpg", pixel_to_mm_ratio=0.5)
print(result.severity.score)
```

### JavaScript
```javascript
import { UDRSClient } from 'udrs-client-js';

const client = new UDRSClient('http://localhost:8000');
const result = await client.analyze('crack.jpg', {
  pixelToMmRatio: 0.5
});
console.log(result.severity.score);
```

---

## Batch Processing

For processing multiple images:

```python
import requests
from pathlib import Path

def batch_analyze(image_paths):
    results = []
    for img_path in image_paths:
        files = {"file": open(img_path, "rb")}
        response = requests.post(
            "http://localhost:8000/api/analyze",
            files=files
        )
        results.append(response.json())
    return results

# Process all images in directory
images = Path("samples").glob("*.jpg")
results = batch_analyze(images)
```

---

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for 4-5x speedup
2. **Skip 3D**: Set `generate_3d=false` for faster processing
3. **Batch Requests**: Use async/parallel requests for multiple images
4. **File Size**: Resize large images (>4K) before upload

---

## Security Best Practices

1. **File Validation**: API enforces file size/type limits
2. **Input Sanitization**: All inputs are validated
3. **CORS**: Configure allowed origins in production
4. **HTTPS**: Use TLS in production (`uvicorn --ssl-keyfile ...`)
5. **API Keys**: Implement authentication for public deployments
