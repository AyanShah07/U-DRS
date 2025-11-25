"""
Volumetric Measurements (3D)
Calculates depth, volume, surface deformation from 3D data
"""

import numpy as np
import open3d as o3d
from typing import Dict, Optional, Tuple
from scipy.spatial import ConvexHull


class VolumetricMeasurements:
    """
    3D volumetric measurements from depth maps and meshes.
    """
    
    def __init__(self, depth_scale: float = 1.0):
        """
        Initialize volumetric calculator.
        
        Args:
            depth_scale: Scale factor for depth values (pixels to mm)
        """
        self.depth_scale = depth_scale
    
    def calculate_max_depth(
        self,
        depth_map: np.ndarray,
        mask: Optional[np.ndarray] = None,
        reference_plane: str = "min"
    ) -> Dict[str, float]:
        """
        Calculate maximum depth of damage.
        
        Args:
            depth_map: Depth map
            mask: Binary mask of damage region
            reference_plane: Reference plane ('min', 'max', 'mean')
            
        Returns:
            Dictionary with max, mean, std depth
        """
        if mask is not None:
            depth_values = depth_map[mask > 0]
        else:
            depth_values = depth_map.flatten()
        
        if len(depth_values) == 0:
            return {"max_mm": 0.0, "mean_mm": 0.0, "std_mm": 0.0, "range_mm": 0.0}
        
        # Determine reference plane
        if reference_plane == "min":
            ref = np.min(depth_values)
        elif reference_plane == "max":
            ref = np.max(depth_values)
        elif reference_plane == "mean":
            ref = np.mean(depth_values)
        else:
            ref = 0.0
        
        # Calculate depths relative to reference
        depths_relative = np.abs(depth_values - ref)
        
        max_depth = np.max(depths_relative) * self.depth_scale
        mean_depth = np.mean(depths_relative) * self.depth_scale
        std_depth = np.std(depths_relative) * self.depth_scale
        range_depth = (np.max(depth_values) - np.min(depth_values)) * self.depth_scale
        
        return {
            "max_mm": float(max_depth),
            "mean_mm": float(mean_depth),
            "std_mm": float(std_depth),
            "range_mm": float(range_depth)
        }
    
    def calculate_volume_from_depth(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        pixel_area_mm2: float = 1.0,
        reference_depth: Optional[float] = None
    ) -> float:
        """
        Calculate volume by integrating depth map.
        
        Args:
            depth_map: Depth map
            mask: Binary damage mask
            pixel_area_mm2: Area of one pixel in mm²
            reference_depth: Reference depth level (auto-compute if None)
            
        Returns:
            Volume in mm³
        """
        masked_depth = depth_map[mask > 0]
        
        if len(masked_depth) == 0:
            return 0.0
        
        # Reference depth (e.g., surrounding surface level)
        if reference_depth is None:
            reference_depth = np.min(masked_depth)
        
        # Volume = sum of (depth - reference) * pixel_area
        depth_diff = np.abs(masked_depth - reference_depth) * self.depth_scale
        volume = np.sum(depth_diff) * pixel_area_mm2
        
        return float(volume)
    
    def calculate_volume_from_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh
    ) -> float:
        """
        Calculate mesh volume.
        
        Args:
            mesh: Triangle mesh
            
        Returns:
            Volume in mm³ (if mesh is watertight)
        """
        if not mesh.is_watertight():
            return 0.0
        
        volume = mesh.get_volume()
        return float(abs(volume))
    
    def calculate_surface_area(
        self,
        mesh: o3d.geometry.TriangleMesh
    ) -> float:
        """
        Calculate mesh surface area.
        
        Args:
            mesh: Triangle mesh
            
        Returns:
            Surface area in mm²
        """
        area = mesh.get_surface_area()
        return float(area)
    
    def calculate_deformation_metrics(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        fit_plane: bool = True
    ) -> Dict[str, float]:
        """
        Calculate surface deformation metrics.
        
        Args:
            depth_map: Depth map
            mask: Damage mask
            fit_plane: Fit reference plane to surrounding area
            
        Returns:
            Dictionary of deformation metrics
        """
        if np.sum(mask) == 0:
            return {
                "deformation_volume_mm3": 0.0,
                "mean_deformation_mm": 0.0,
                "max_deformation_mm": 0.0
            }
        
        # Get coordinates
        coords = np.column_stack(np.where(mask > 0))
        depth_values = depth_map[mask > 0]
        
        if fit_plane:
            # Fit plane to surrounding area (inverse mask)
            surround_mask = 1 - mask
            kernel = np.ones((15, 15), np.uint8)
            import cv2
            surround_mask = cv2.dilate(surround_mask.astype(np.uint8), kernel, iterations=1)
            surround_mask = surround_mask - (1 - mask)
            
            if np.sum(surround_mask) > 100:
                surround_coords = np.column_stack(np.where(surround_mask > 0))
                surround_depths = depth_map[surround_mask > 0]
                
                # Fit plane: z = ax + by + c
                A = np.column_stack([surround_coords, np.ones(len(surround_coords))])
                plane_params, _, _, _ = np.linalg.lstsq(A, surround_depths, rcond=None)
                
                # Compute reference depths at damage locations
                A_damage = np.column_stack([coords, np.ones(len(coords))])
                reference_depths = A_damage @ plane_params
            else:
                # Fallback to mean depth
                reference_depths = np.mean(depth_values)
        else:
            reference_depths = np.min(depth_values)
        
        # Deformation = actual depth - reference depth
        deformations = np.abs(depth_values - reference_depths) * self.depth_scale
        
        # Volume (simplified: sum of deformations * unit area)
        pixel_area = 1.0  # Assumes calibrated units
        deformation_volume = np.sum(deformations) * pixel_area
        
        return {
            "deformation_volume_mm3": float(deformation_volume),
            "mean_deformation_mm": float(np.mean(deformations)),
            "max_deformation_mm": float(np.max(deformations))
        }
    
    def calculate_all(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        mesh: Optional[o3d.geometry.TriangleMesh] = None,
        pixel_area_mm2: float = 1.0
    ) -> Dict[str, any]:
        """
        Calculate all volumetric measurements.
        
        Args:
            depth_map: Depth map
            mask: Damage mask
            mesh: Optional mesh
            pixel_area_mm2: Pixel area for volume calculation
            
        Returns:
            Dictionary of all measurements
        """
        # Depth measurements
        depth_stats = self.calculate_max_depth(depth_map, mask)
        
        # Volume from depth
        volume_depth = self.calculate_volume_from_depth(
            depth_map, mask, pixel_area_mm2
        )
        
        # Deformation metrics
        deformation = self.calculate_deformation_metrics(depth_map, mask)
        
        results = {
            "depth_stats": depth_stats,
            "volume_from_depth_mm3": volume_depth,
            "deformation": deformation
        }
        
        # Mesh measurements if available
        if mesh is not None:
            results["volume_from_mesh_mm3"] = self.calculate_volume_from_mesh(mesh)
            results["surface_area_mm2"] = self.calculate_surface_area(mesh)
        
        return results


def create_volumetric_calculator(depth_scale: float = 1.0) -> VolumetricMeasurements:
    """
    Factory function to create volumetric calculator.
    
    Args:
        depth_scale: Depth scale factor
        
    Returns:
        VolumetricMeasurements instance
    """
    return VolumetricMeasurements(depth_scale=depth_scale)
