"""
Configuration management for U-DRS pipeline.
Centralizes all model paths, hyperparameters, and processing settings.
"""

import os
from pathlib import Path
from typing import Dict, Any
import torch

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, SAMPLES_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class Config:
    """Global configuration for U-DRS system."""
    
    # Base paths
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = DATA_DIR
    MODELS_DIR = MODELS_DIR
    SAMPLES_DIR = SAMPLES_DIR
    OUTPUTS_DIR = OUTPUTS_DIR
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_MIXED_PRECISION = True if DEVICE == "cuda" else False
    
    # Model paths
    DETECTOR_MODEL_PATH = MODELS_DIR / "damage_detector.pth"
    SEGMENTATION_MODEL_PATH = MODELS_DIR / "unet_segmentation.pth"
    DEPTH_MODEL_TYPE = "DPT_Large"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
    SEVERITY_MODEL_PATH = MODELS_DIR / "severity_model.pkl"
    COST_MODEL_PATH = MODELS_DIR / "cost_predictor.pkl"
    
    # Image processing parameters
    INPUT_SIZE = (640, 480)  # Width, Height for processing
    DETECTION_THRESHOLD = 0.5  # Confidence threshold for damage detection
    SEGMENTATION_THRESHOLD = 0.5  # Threshold for binary mask
    
    # Edge detection parameters
    CANNY_LOW_THRESHOLD = 50
    CANNY_HIGH_THRESHOLD = 150
    MORPH_KERNEL_SIZE = 5
    
    # 3D Reconstruction parameters
    CAMERA_FOV = 60.0  # Field of view in degrees
    CAMERA_FOCAL_LENGTH = None  # Auto-compute if None
    POISSON_DEPTH = 9  # Octree depth for Poisson reconstruction
    POINT_CLOUD_VOXEL_SIZE = 0.01  # Voxel size for downsampling
    
    # Measurement parameters
    PIXEL_TO_MM_RATIO = 1.0  # Calibration factor (default: 1 pixel = 1mm)
    MIN_CRACK_LENGTH_MM = 5.0  # Minimum crack length to report
    MIN_DAMAGE_AREA_MM2 = 10.0  # Minimum damage area to report
    
    # Severity scoring thresholds
    SEVERITY_THRESHOLDS = {
        "minor": (0, 25),
        "moderate": (25, 50),
        "severe": (50, 75),
        "critical": (75, 100)
    }
    
    # Cost prediction parameters (placeholder values)
    BASE_REPAIR_COST = 100.0  # USD
    COST_PER_MM2 = 0.5  # USD per mm²
    COST_PER_MM_DEPTH = 2.0  # USD per mm depth
    
    # API settings
    API_HOST = "127.0.0.1"
    API_PORT = 8080
    MAX_UPLOAD_SIZE_MB = 50
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = PROJECT_ROOT / "udrs.log"
    
    # Optimization flags
    ENABLE_ONNX = False  # Set to True to use ONNX runtime
    ENABLE_TENSORRT = False  # Set to True for TensorRT optimization
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and not callable(getattr(cls, key))
        }
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")


# Export singleton instance
config = Config()
