"""
Depth Estimation Module
Integrates MiDaS/DPT for monocular depth estimation
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple
from pathlib import Path
import urllib.request


class DepthEstimator:
    """
    Wrapper for MiDaS depth estimation models.
    """
    
    def __init__(
        self,
        model_type: str = "DPT_Large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        optimize: bool = True
    ):
        """
        Initialize depth estimator.
        
        Args:
            model_type: Model variant (DPT_Large, DPT_Hybrid, MiDaS_small)
            device: Device to run inference on
            optimize: Apply optimization for inference
        """
        self.device = device
        self.model_type = model_type
        
        # Load MiDaS model from torch hub
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
        except Exception as e:
            print(f"Failed to load {model_type}, falling back to MiDaS_small: {e}")
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
            self.model_type = "MiDaS_small"
        
        self.model.to(device)
        self.model.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        # Optimization
        if optimize and device == "cuda":
            self.model = torch.jit.trace(
                self.model,
                torch.randn(1, 3, 384, 384).to(device)
            )
    
    @torch.no_grad()
    def estimate(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth map from image.
        
        Args:
            image: PIL Image
            
        Returns:
            Depth map as numpy array (relative depth, inverse depth)
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Preprocess
        input_batch = self.transform(image).to(self.device)
        
        # Inference
        prediction = self.model(input_batch)
        
        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],  # (height, width)
            mode="bicubic",
            align_corners=False
        ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        return depth_map
    
    def estimate_from_path(self, image_path: Path) -> np.ndarray:
        """
        Estimate depth from image file.
        
        Args:
            image_path: Path to image
            
        Returns:
            Depth map
        """
        image = Image.open(image_path)
        return self.estimate(image)
    
    def normalize_depth(self, depth_map: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Normalize depth map to [0, 1] range.
        
        Args:
            depth_map: Raw depth map
            method: Normalization method (minmax, percentile)
            
        Returns:
            Normalized depth map
        """
        if method == "minmax":
            d_min, d_max = depth_map.min(), depth_map.max()
            normalized = (depth_map - d_min) / (d_max - d_min + 1e-8)
        elif method == "percentile":
            p2, p98 = np.percentile(depth_map, [2, 98])
            normalized = np.clip((depth_map - p2) / (p98 - p2 + 1e-8), 0, 1)
        else:
            normalized = depth_map
        
        return normalized
    
    def align_depth_with_mask(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extract depth values only in masked region.
        
        Args:
            depth_map: Depth map
            mask: Binary segmentation mask
            normalize: Normalize depth in masked region
            
        Returns:
            Aligned depth map (0 outside mask)
        """
        masked_depth = depth_map * mask
        
        if normalize and np.sum(mask) > 0:
            # Normalize only within masked region
            masked_values = masked_depth[mask > 0]
            d_min, d_max = masked_values.min(), masked_values.max()
            masked_depth[mask > 0] = (masked_values - d_min) / (d_max - d_min + 1e-8)
        
        return masked_depth
    
    def get_depth_statistics(self, depth_map: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
        """
        Calculate depth statistics.
        
        Args:
            depth_map: Depth map
            mask: Optional binary mask to compute stats only in masked region
            
        Returns:
            Dictionary of statistics
        """
        if mask is not None:
            values = depth_map[mask > 0]
        else:
            values = depth_map.flatten()
        
        if len(values) == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "range": 0.0
            }
        
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values))
        }


def create_depth_estimator(
    model_type: str = "DPT_Large",
    device: str = "cuda"
) -> DepthEstimator:
    """
    Factory function to create depth estimator.
    
    Args:
        model_type: Model variant
        device: Device to use
        
    Returns:
        DepthEstimator instance
    """
    return DepthEstimator(model_type=model_type, device=device)
