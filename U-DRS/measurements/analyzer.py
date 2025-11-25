"""
Unified Measurement Analyzer
Combines 2D geometric and 3D volumetric measurements
"""

import numpy as np
from typing import Dict, Optional, List
import open3d as o3d

from measurements.geometric import GeometricMeasurements
from measurements.volumetric import VolumetricMeasurements


class DamageAnalyzer:
    """
    Unified interface for all damage measurements.
    """
    
    def __init__(
        self,
        pixel_to_mm_ratio: float = 1.0,
        depth_scale: float = 1.0
    ):
        """
        Initialize analyzer.
        
        Args:
            pixel_to_mm_ratio: 2D calibration ratio
            depth_scale: 3D depth scale factor
        """
        self.geometric = GeometricMeasurements(pixel_to_mm_ratio)
        self.volumetric = VolumetricMeasurements(depth_scale)
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        self.depth_scale = depth_scale
    
    def analyze_complete(
        self,
        mask: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        contours: Optional[List[np.ndarray]] = None,
        mesh: Optional[o3d.geometry.TriangleMesh] = None
    ) -> Dict[str, any]:
        """
        Perform complete damage analysis.
        
        Args:
            mask: Binary segmentation mask
            depth_map: Optional depth map for 3D measurements
            contours: Optional precomputed contours
            mesh: Optional 3D mesh
            
        Returns:
            Complete measurement report
        """
        report = {
            "2d_measurements": {},
            "3d_measurements": {},
            "summary": {}
        }
        
        # 2D geometric measurements
        geometric_results = self.geometric.calculate_all(mask, contours)
        report["2d_measurements"] = geometric_results
        
        # 3D volumetric measurements
        if depth_map is not None:
            pixel_area_mm2 = self.pixel_to_mm_ratio ** 2
            volumetric_results = self.volumetric.calculate_all(
                depth_map, mask, mesh, pixel_area_mm2
            )
            report["3d_measurements"] = volumetric_results
        
        # Generate summary
        report["summary"] = self._generate_summary(report)
        
        return report
    
    def _generate_summary(self, report: Dict) -> Dict[str, any]:
        """
        Generate human-readable summary.
        
        Args:
            report: Full measurement report
            
        Returns:
            Summary dictionary
        """
        summary = {}
        
        # Extract key metrics
        geom = report["2d_measurements"]
        summary["damage_area_mm2"] = geom["area_mm2"]
        summary["crack_length_mm"] = geom["crack_length_mm"]
        summary["crack_width_mean_mm"] = geom["crack_width"]["mean"]
        summary["crack_width_max_mm"] = geom["crack_width"]["max"]
        summary["bbox_width_mm"] = geom["bounding_box"]["width"]
        summary["bbox_height_mm"] = geom["bounding_box"]["height"]
        
        # 3D metrics if available
        if "3d_measurements" in report and report["3d_measurements"]:
            vol = report["3d_measurements"]
            if "depth_stats" in vol:
                summary["max_depth_mm"] = vol["depth_stats"]["max_mm"]
                summary["mean_depth_mm"] = vol["depth_stats"]["mean_mm"]
            if "volume_from_depth_mm3" in vol:
                summary["volume_mm3"] = vol["volume_from_depth_mm3"]
            if "deformation" in vol:
                summary["max_deformation_mm"] = vol["deformation"]["max_deformation_mm"]
        
        return summary
    
    def get_damage_severity_features(self, report: Dict) -> np.ndarray:
        """
        Extract feature vector for severity prediction.
        
        Args:
            report: Measurement report
            
        Returns:
            Feature array
        """
        features = []
        
        # 2D features
        geom = report["2d_measurements"]
        features.extend([
            geom["area_mm2"],
            geom["perimeter_mm"],
            geom["crack_length_mm"],
            geom["crack_width"]["mean"],
            geom["crack_width"]["max"],
            geom["bounding_box"]["width"],
            geom["bounding_box"]["height"],
            geom["bounding_box"]["aspect_ratio"],
            geom["compactness"],
            geom["circularity"],
            geom["num_regions"]
        ])
        
        # 3D features (use 0 if not available)
        if "3d_measurements" in report and report["3d_measurements"]:
            vol = report["3d_measurements"]
            if "depth_stats" in vol:
                features.extend([
                    vol["depth_stats"]["max_mm"],
                    vol["depth_stats"]["mean_mm"],
                    vol["depth_stats"]["std_mm"],
                    vol["depth_stats"]["range_mm"]
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            if "deformation" in vol:
                features.extend([
                    vol["deformation"]["deformation_volume_mm3"],
                    vol["deformation"]["mean_deformation_mm"],
                    vol["deformation"]["max_deformation_mm"]
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            # Pad with zeros for missing 3D features
            features.extend([0.0] * 7)
        
        return np.array(features, dtype=np.float32)
    
    def compare_with_thresholds(
        self,
        report: Dict,
        thresholds: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Compare measurements against thresholds.
        
        Args:
            report: Measurement report
            thresholds: Dictionary of threshold values
            
        Returns:
            Dictionary of pass/fail flags
        """
        summary = report["summary"]
        results = {}
        
        for key, threshold in thresholds.items():
            if key in summary:
                results[key] = summary[key] > threshold
        
        return results


def create_analyzer(
    pixel_to_mm_ratio: float = 1.0,
    depth_scale: float = 1.0
) -> DamageAnalyzer:
    """
    Factory function to create analyzer.
    
    Args:
        pixel_to_mm_ratio: 2D calibration
        depth_scale: 3D calibration
        
    Returns:
        DamageAnalyzer instance
    """
    return DamageAnalyzer(pixel_to_mm_ratio, depth_scale)
