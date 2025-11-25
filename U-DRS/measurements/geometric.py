"""
Geometric Measurements (2D)
Calculates crack length, width, area, perimeter from segmentation masks
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
from scipy import ndimage
from skimage.morphology import skeletonize


class GeometricMeasurements:
    """
    2D geometric measurements from segmentation masks.
    """
    
    def __init__(self, pixel_to_mm_ratio: float = 1.0):
        """
        Initialize measurement calculator.
        
        Args:
            pixel_to_mm_ratio: Calibration ratio (pixels to mm)
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
    
    def calculate_area(self, mask: np.ndarray) -> float:
        """
        Calculate damaged area.
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Area in mm²
        """
        num_pixels = np.sum(mask > 0)
        area_mm2 = num_pixels * (self.pixel_to_mm_ratio ** 2)
        return float(area_mm2)
    
    def calculate_perimeter(self, contour: np.ndarray) -> float:
        """
        Calculate contour perimeter.
        
        Args:
            contour: Contour array
            
        Returns:
            Perimeter in mm
        """
        perimeter_pixels = cv2.arcLength(contour, closed=True)
        perimeter_mm = perimeter_pixels * self.pixel_to_mm_ratio
        return float(perimeter_mm)
    
    def calculate_bounding_box(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate bounding box dimensions.
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary with width, height, area
        """
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            return {"width": 0.0, "height": 0.0, "area": 0.0, "aspect_ratio": 0.0}
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        width_pixels = x_max - x_min + 1
        height_pixels = y_max - y_min + 1
        
        width_mm = width_pixels * self.pixel_to_mm_ratio
        height_mm = height_pixels * self.pixel_to_mm_ratio
        area_mm2 = width_mm * height_mm
        aspect_ratio = width_mm / height_mm if height_mm > 0 else 0.0
        
        return {
            "width": float(width_mm),
            "height": float(height_mm),
            "area": float(area_mm2),
            "aspect_ratio": float(aspect_ratio),
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max)
        }
    
    def calculate_skeleton_length(self, mask: np.ndarray) -> float:
        """
        Calculate crack length using morphological skeleton.
        
        Args:
            mask: Binary mask
            
        Returns:
            Length in mm
        """
        # Skeletonize
        skeleton = skeletonize(mask > 0)
        
        # Count skeleton pixels (approximation of length)
        num_pixels = np.sum(skeleton)
        length_mm = num_pixels * self.pixel_to_mm_ratio
        
        return float(length_mm)
    
    def calculate_crack_width(
        self,
        mask: np.ndarray,
        skeleton: Optional[np.ndarray] = None,
        num_samples: int = 20
    ) -> Dict[str, float]:
        """
        Calculate crack width statistics.
        
        Args:
            mask: Binary damage mask
            skeleton: Precomputed skeleton (computed if None)
            num_samples: Number of sample points along skeleton
            
        Returns:
            Dictionary with mean, max, min, std width
        """
        if skeleton is None:
            skeleton = skeletonize(mask > 0)
        
        # Get skeleton points
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        
        if len(skeleton_points) < num_samples:
            num_samples = len(skeleton_points)
        
        if num_samples == 0:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
        
        # Sample uniformly
        indices = np.linspace(0, len(skeleton_points) - 1, num_samples, dtype=int)
        sampled_points = skeleton_points[indices]
        
        # Compute distance transform
        dist_transform = ndimage.distance_transform_edt(mask)
        
        # Width = 2 * distance to nearest boundary
        widths = []
        for point in sampled_points:
            width_pixels = dist_transform[point[0], point[1]] * 2
            width_mm = width_pixels * self.pixel_to_mm_ratio
            widths.append(width_mm)
        
        widths = np.array(widths)
        
        return {
            "mean": float(np.mean(widths)),
            "max": float(np.max(widths)),
            "min": float(np.min(widths)),
            "std": float(np.std(widths))
        }
    
    def calculate_compactness(self, area: float, perimeter: float) -> float:
        """
        Calculate shape compactness (4π × area / perimeter²).
        Circle = 1.0, more irregular shapes < 1.0
        
        Args:
            area: Shape area
            perimeter: Shape perimeter
            
        Returns:
            Compactness score
        """
        if perimeter == 0:
            return 0.0
        
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        return float(compactness)
    
    def calculate_circularity(self, contour: np.ndarray) -> float:
        """
        Calculate circularity of contour.
        
        Args:
            contour: Contour array
            
        Returns:
            Circularity score (0-1, 1 = perfect circle)
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return float(np.clip(circularity, 0, 1))
    
    def calculate_all(
        self,
        mask: np.ndarray,
        contours: Optional[List[np.ndarray]] = None
    ) -> Dict[str, any]:
        """
        Calculate all geometric measurements.
        
        Args:
            mask: Binary segmentation mask
            contours: Optional precomputed contours
            
        Returns:
            Dictionary of all measurements
        """
        # Find contours if not provided
        if contours is None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        
        # Get largest contour
        if len(contours) == 0:
            return {
                "area_mm2": 0.0,
                "perimeter_mm": 0.0,
                "crack_length_mm": 0.0,
                "crack_width": {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0},
                "bounding_box": {"width": 0.0, "height": 0.0, "area": 0.0, "aspect_ratio": 0.0},
                "compactness": 0.0,
                "circularity": 0.0,
                "num_regions": 0
            }
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate measurements
        area = self.calculate_area(mask)
        perimeter = self.calculate_perimeter(largest_contour)
        crack_length = self.calculate_skeleton_length(mask)
        skeleton = skeletonize(mask > 0)
        crack_width = self.calculate_crack_width(mask, skeleton)
        bbox = self.calculate_bounding_box(mask)
        compactness = self.calculate_compactness(area, perimeter)
        circularity = self.calculate_circularity(largest_contour)
        
        return {
            "area_mm2": area,
            "perimeter_mm": perimeter,
            "crack_length_mm": crack_length,
            "crack_width": crack_width,
            "bounding_box": bbox,
            "compactness": compactness,
            "circularity": circularity,
            "num_regions": len(contours)
        }


def create_geometric_calculator(pixel_to_mm_ratio: float = 1.0) -> GeometricMeasurements:
    """
    Factory function to create geometric calculator.
    
    Args:
        pixel_to_mm_ratio: Calibration ratio
        
    Returns:
        GeometricMeasurements instance
    """
    return GeometricMeasurements(pixel_to_mm_ratio=pixel_to_mm_ratio)
