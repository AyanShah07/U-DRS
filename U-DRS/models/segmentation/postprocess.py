"""
Post-processing utilities for segmentation masks.
Includes morphological operations, edge detection, and contour extraction.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage


class SegmentationPostProcessor:
    """
    Post-processing for segmentation masks.
    """
    
    def __init__(
        self,
        morph_kernel_size: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        min_contour_area: int = 100
    ):
        """
        Initialize post-processor.
        
        Args:
            morph_kernel_size: Kernel size for morphological operations
            canny_low: Low threshold for Canny edge detection
            canny_high: High threshold for Canny edge detection
            min_contour_area: Minimum contour area to keep
        """
        self.morph_kernel_size = morph_kernel_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area
        
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size)
        )
    
    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean binary mask using morphological operations.
        
        Args:
            mask: Binary mask (0/1)
            
        Returns:
            Cleaned mask
        """
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Morphological opening (remove noise)
        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, self.kernel)
        
        # Morphological closing (fill holes)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)
        
        # Convert back to binary
        return (closed > 127).astype(np.uint8)
    
    def extract_edges(self, mask: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract edges from mask or image.
        
        Args:
            mask: Binary mask
            image: Optional grayscale image for edge detection
            
        Returns:
            Edge map
        """
        if image is not None:
            # Use Canny on original image, masked by segmentation
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            # Mask edges with segmentation
            edges = edges * mask
        else:
            # Use mask edges
            mask_uint8 = (mask * 255).astype(np.uint8)
            edges = cv2.Canny(mask_uint8, self.canny_low, self.canny_high)
        
        return edges
    
    def find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in binary mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            List of contours
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter by area
        filtered_contours = [
            cnt for cnt in contours
            if cv2.contourArea(cnt) >= self.min_contour_area
        ]
        
        return filtered_contours
    
    def get_largest_contour(self, contours: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Get the largest contour by area.
        
        Args:
            contours: List of contours
            
        Returns:
            Largest contour or None
        """
        if not contours:
            return None
        
        return max(contours, key=cv2.contourArea)
    
    def approximate_polygon(self, contour: np.ndarray, epsilon_factor: float = 0.01) -> np.ndarray:
        """
        Approximate contour as polygon.
        
        Args:
            contour: Input contour
            epsilon_factor: Approximation accuracy factor
            
        Returns:
            Approximated polygon
        """
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx
    
    def skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute morphological skeleton of mask (useful for crack length).
        
        Args:
            mask: Binary mask
            
        Returns:
            Skeleton mask
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        skeleton = np.zeros_like(mask_uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(mask_uint8, opened)
            eroded = cv2.erode(mask_uint8, element)
            skeleton = cv2.bitwise_or(skeleton, temp)
            mask_uint8 = eroded.copy()
            
            if cv2.countNonZero(mask_uint8) == 0:
                break
        
        return (skeleton > 0).astype(np.uint8)
    
    def get_skeleton_length(self, skeleton: np.ndarray, pixel_to_mm_ratio: float = 1.0) -> float:
        """
        Calculate skeleton length (useful for crack length).
        
        Args:
            skeleton: Binary skeleton mask
            pixel_to_mm_ratio: Conversion ratio
            
        Returns:
            Length in mm
        """
        # Count skeleton pixels
        num_pixels = np.sum(skeleton)
        length_mm = num_pixels * pixel_to_mm_ratio
        return length_mm
    
    def get_crack_width(
        self,
        mask: np.ndarray,
        skeleton: np.ndarray,
        pixel_to_mm_ratio: float = 1.0,
        num_samples: int = 10
    ) -> Tuple[float, float, float]:
        """
        Estimate crack width by measuring perpendicular distances.
        
        Args:
            mask: Binary damage mask
            skeleton: Skeleton of the mask
            pixel_to_mm_ratio: Conversion ratio
            num_samples: Number of points to sample along skeleton
            
        Returns:
            Tuple of (mean_width, max_width, min_width) in mm
        """
        # Get skeleton points
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        
        if len(skeleton_points) < num_samples:
            num_samples = len(skeleton_points)
        
        if num_samples == 0:
            return 0.0, 0.0, 0.0
        
        # Sample points uniformly
        indices = np.linspace(0, len(skeleton_points) - 1, num_samples, dtype=int)
        sampled_points = skeleton_points[indices]
        
        # Compute distance transform of mask
        dist_transform = ndimage.distance_transform_edt(mask)
        
        widths = []
        for point in sampled_points:
            # Width is 2 * distance to nearest boundary
            width_pixels = dist_transform[point[0], point[1]] * 2
            width_mm = width_pixels * pixel_to_mm_ratio
            widths.append(width_mm)
        
        widths = np.array(widths)
        return float(np.mean(widths)), float(np.max(widths)), float(np.min(widths))
    
    def process(self, mask: np.ndarray, image: Optional[np.ndarray] = None) -> dict:
        """
        Complete post-processing pipeline.
        
        Args:
            mask: Binary segmentation mask
            image: Optional original image
            
        Returns:
            Dictionary containing all processed outputs
        """
        # Clean mask
        cleaned_mask = self.clean_mask(mask)
        
        # Extract edges
        edges = self.extract_edges(cleaned_mask, image)
        
        # Find contours
        contours = self.find_contours(cleaned_mask)
        
        # Get skeleton
        skeleton = self.skeletonize(cleaned_mask)
        
        return {
            "cleaned_mask": cleaned_mask,
            "edges": edges,
            "contours": contours,
            "skeleton": skeleton,
            "num_contours": len(contours),
            "largest_contour": self.get_largest_contour(contours)
        }


def create_postprocessor(
    morph_kernel_size: int = 5,
    canny_low: int = 50,
    canny_high: int = 150
) -> SegmentationPostProcessor:
    """
    Factory function to create postprocessor.
    
    Args:
        morph_kernel_size: Kernel size for morphological operations
        canny_low: Low threshold for Canny
        canny_high: High threshold for Canny
        
    Returns:
        SegmentationPostProcessor instance
    """
    return SegmentationPostProcessor(
        morph_kernel_size=morph_kernel_size,
        canny_low=canny_low,
        canny_high=canny_high
    )
