"""
Depth utilities for visualization and metric calibration.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, Tuple


def visualize_depth(
    depth_map: np.ndarray,
    colormap: str = "viridis",
    invert: bool = True
) -> np.ndarray:
    """
    Visualize depth map with colormap.
    
    Args:
        depth_map: Depth map array
        colormap: Matplotlib colormap name
        invert: Invert depth (closer = brighter)
        
    Returns:
        RGB visualization image
    """
    # Normalize to [0, 1]
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    
    if invert:
        depth_norm = 1.0 - depth_norm
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(depth_norm)
    
    # Convert to RGB uint8
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return rgb


def create_depth_overlay(
    image: np.ndarray,
    depth_map: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Create overlay of depth map on original image.
    
    Args:
        image: Original RGB image
        depth_map: Depth map
        alpha: Blending factor
        colormap: Colormap for depth
        
    Returns:
        Blended image
    """
    depth_vis = visualize_depth(depth_map, colormap=colormap)
    
    # Ensure same size
    if image.shape[:2] != depth_vis.shape[:2]:
        depth_vis = cv2.resize(depth_vis, (image.shape[1], image.shape[0]))
    
    # Blend
    blended = cv2.addWeighted(image, 1 - alpha, depth_vis, alpha, 0)
    
    return blended


def calibrate_depth_with_reference(
    depth_map: np.ndarray,
    reference_points: list,
    real_distances: list
) -> Tuple[np.ndarray, float]:
    """
    Calibrate relative depth to metric depth using reference points.
    
    Args:
        depth_map: Relative depth map from MiDaS
        reference_points: List of (y, x) coordinates of known points
        real_distances: List of real-world distances (in mm) from camera
        
    Returns:
        Tuple of (calibrated_depth_map, scale_factor)
    """
    # Extract depth values at reference points
    depth_values = [depth_map[y, x] for y, x in reference_points]
    
    # Fit linear relationship: real_distance = scale * depth_value + offset
    # For simplicity, we assume offset = 0 and just compute scale
    scale_factor = np.mean([real / depth for real, depth in zip(real_distances, depth_values)])
    
    calibrated_depth = depth_map * scale_factor
    
    return calibrated_depth, scale_factor


def compute_depth_gradient(depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute depth gradient (useful for detecting sharp changes).
    
    Args:
        depth_map: Depth map
        
    Returns:
        Tuple of (gradient_magnitude, gradient_direction)
    """
    # Compute gradients using Sobel
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    
    return magnitude, direction


def estimate_surface_normals(depth_map: np.ndarray, fx: float = 525.0, fy: float = 525.0) -> np.ndarray:
    """
    Estimate surface normals from depth map.
    
    Args:
        depth_map: Depth map
        fx, fy: Camera focal lengths (pixels)
        
    Returns:
        Normal map (H x W x 3)
    """
    h, w = depth_map.shape
    
    # Create meshgrid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to 3D points (assuming principal point at center)
    cx, cy = w / 2, h / 2
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    
    # Compute gradients
    dzdx = np.gradient(z, axis=1)
    dzdy = np.gradient(z, axis=0)
    
    # Normal vectors
    normals = np.stack([-dzdx, -dzdy, np.ones_like(z)], axis=-1)
    
    # Normalize
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norms + 1e-8)
    
    return normals


def detect_depth_discontinuities(
    depth_map: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Detect depth discontinuities (edges in depth).
    
    Args:
        depth_map: Depth map
        threshold: Gradient threshold for discontinuity
        
    Returns:
        Binary mask of discontinuities
    """
    gradient_mag, _ = compute_depth_gradient(depth_map)
    
    # Normalize gradient
    if gradient_mag.max() > 0:
        gradient_mag = gradient_mag / gradient_mag.max()
    
    # Threshold
    discontinuities = (gradient_mag > threshold).astype(np.uint8)
    
    return discontinuities
