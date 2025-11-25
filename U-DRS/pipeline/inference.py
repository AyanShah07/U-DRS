"""
End-to-End Inference Pipeline
Orchestrates all modules from detection to final report generation
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import time
import torch

from pipeline.config import config
from utils.logger import setup_logger, get_logger
from models.detection.damage_detector import DamageDetectorInference
from models.segmentation.unet_model import DamageSegmenter
from models.segmentation.postprocess import SegmentationPostProcessor
from models.depth.depth_estimator import DepthEstimator
from reconstruction.point_cloud import PointCloudGenerator
from reconstruction.mesh_builder import MeshBuilder
from measurements.analyzer import DamageAnalyzer
from models.prediction.severity_model import SeverityScorer
from models.prediction.cost_predictor import CostPredictor


# Setup logger
logger = setup_logger(config.LOG_FILE, config.LOG_LEVEL)


class UDRSPipeline:
    """
    Universal Damage Reconstruction System - Complete Pipeline
    """
    
    def __init__(
        self,
        device: str = None,
        pixel_to_mm_ratio: float = 1.0,
        depth_scale: float = 1.0
    ):
        """
        Initialize U-DRS pipeline.
        
        Args:
            device: Device to run on (cuda/cpu)
            pixel_to_mm_ratio: Calibration ratio for 2D measurements
            depth_scale: Calibration for depth measurements
        """
        self.device = device or config.DEVICE
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        self.depth_scale = depth_scale
        
        logger.info(f"Initializing U-DRS Pipeline on {self.device}")
        
        # Initialize modules (lazy loading)
        self.detector = None
        self.segmenter = None
        self.postprocessor = None
        self.depth_estimator = None
        self.pcd_generator = None
        self.mesh_builder = None
        self.analyzer = None
        self.severity_scorer = None
        self.cost_predictor = None
        
        self._load_modules()
        
        logger.info("U-DRS Pipeline initialized successfully")
    
    def _load_modules(self):
        """Load all pipeline modules."""
        logger.info("Loading pipeline modules...")
        
        # Detection
        self.detector = DamageDetectorInference(
            model_path=config.DETECTOR_MODEL_PATH if config.DETECTOR_MODEL_PATH.exists() else None,
            device=self.device,
            threshold=config.DETECTION_THRESHOLD
        )
        
        # Segmentation
        self.segmenter = DamageSegmenter(
            model_path=config.SEGMENTATION_MODEL_PATH if config.SEGMENTATION_MODEL_PATH.exists() else None,
            device=self.device,
            threshold=config.SEGMENTATION_THRESHOLD,
            input_size=config.INPUT_SIZE
        )
        
        # Post-processing
        self.postprocessor = SegmentationPostProcessor(
            morph_kernel_size=config.MORPH_KERNEL_SIZE,
            canny_low=config.CANNY_LOW_THRESHOLD,
            canny_high=config.CANNY_HIGH_THRESHOLD
        )
        
        # Depth estimation
        self.depth_estimator = DepthEstimator(
            model_type=config.DEPTH_MODEL_TYPE,
            device=self.device
        )
        
        # 3D reconstruction
        self.pcd_generator = PointCloudGenerator(fov=config.CAMERA_FOV)
        self.mesh_builder = MeshBuilder(method="poisson")
        
        # Measurements
        self.analyzer = DamageAnalyzer(
            pixel_to_mm_ratio=self.pixel_to_mm_ratio,
            depth_scale=self.depth_scale
        )
        
        # Predictions
        self.severity_scorer = SeverityScorer(
            model_path=config.SEVERITY_MODEL_PATH if config.SEVERITY_MODEL_PATH.exists() else None,
            use_ml=False  # Start with rule-based
        )
        
        self.cost_predictor = CostPredictor(
            cost_model_path=config.COST_MODEL_PATH if config.COST_MODEL_PATH.exists() else None,
            base_cost=config.BASE_REPAIR_COST
        )
    
    def process(
        self,
        image_path: Union[str, Path],
        save_outputs: bool = True,
        output_dir: Optional[Path] = None,
        generate_3d: bool = True
    ) -> Dict:
        """
        Process single image through complete pipeline.
        
        Args:
            image_path: Path to input image
            save_outputs: Save intermediate results
            output_dir: Output directory for results
            generate_3d: Generate 3D reconstruction
            
        Returns:
            Complete analysis result dictionary
        """
        image_path = Path(image_path)
        logger.info(f"Processing image: {image_path.name}")
        
        if output_dir is None:
            output_dir = config.OUTPUTS_DIR / image_path.stem
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track timing
        timing = {}
        start_total = time.time()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Step 1: Damage Detection
        logger.info("Step 1: Damage Detection")
        start = time.time()
        is_damaged, confidence, probs = self.detector.detect(image)
        timing["detection"] = time.time() - start
        
        if not is_damaged:
            logger.info(f"No damage detected (confidence: {confidence:.2%})")
            return {
                "status": "no_damage",
                "confidence": confidence,
                "timing": timing
            }
        
        logger.info(f"Damage detected with {confidence:.2%} confidence")
        
        # Step 2: Segmentation
        logger.info("Step 2: Damage Segmentation")
        start = time.time()
        binary_mask, prob_map = self.segmenter.segment(image)
        timing["segmentation"] = time.time() - start
        
        # Step 3: Post-processing
        logger.info("Step 3: Post-processing")
        start = time.time()
        postprocess_results = self.postprocessor.process(binary_mask, image_np)
        cleaned_mask = postprocess_results["cleaned_mask"]
        contours = postprocess_results["contours"]
        skeleton = postprocess_results["skeleton"]
        edges = postprocess_results["edges"]
        timing["postprocessing"] = time.time() - start
        
        # Step 4: Depth Estimation
        logger.info("Step 4: Depth Estimation")
        start = time.time()
        depth_map = self.depth_estimator.estimate(image)
        depth_map_normalized = self.depth_estimator.normalize_depth(depth_map)
        aligned_depth = self.depth_estimator.align_depth_with_mask(
            depth_map_normalized,
            cleaned_mask,
            normalize=True
        )
        timing["depth_estimation"] = time.time() - start
        
        # Step 5: 3D Reconstruction
        mesh = None
        pcd = None
        if generate_3d:
            logger.info("Step 5: 3D Reconstruction")
            start = time.time()
            
            # Generate point cloud
            pcd = self.pcd_generator.depth_to_pointcloud(
                aligned_depth,
                rgb_image=image_np,
                mask=cleaned_mask,
                depth_scale=self.depth_scale
            )
            
            # Filter point cloud
            pcd = self.pcd_generator.filter_pointcloud(
                pcd,
                voxel_size=config.POINT_CLOUD_VOXEL_SIZE
            )
            
            # Estimate normals
            pcd = self.pcd_generator.estimate_normals(pcd)
            
            # Build mesh
            try:
                mesh = self.mesh_builder.build(
                    pcd,
                    depth=config.POISSON_DEPTH
                )
                mesh = self.mesh_builder.refine_mesh(mesh, iterations=1)
            except Exception as e:
                logger.warning(f"Mesh reconstruction failed: {e}")
                mesh = None
            
            timing["3d_reconstruction"] = time.time() - start
        
        # Step 6: Measurements
        logger.info("Step 6: Calculating Measurements")
        start = time.time()
        measurement_report = self.analyzer.analyze_complete(
            mask=cleaned_mask,
            depth_map=aligned_depth,
            contours=contours,
            mesh=mesh
        )
        timing["measurements"] = time.time() - start
        
        # Step 7: Severity Scoring
        logger.info("Step 7: Severity Scoring")
        start = time.time()
        features = self.analyzer.get_damage_severity_features(measurement_report)
        severity_result = self.severity_scorer.predict(features)
        timing["severity_scoring"] = time.time() - start
        
        # Step 8: Cost & Urgency Prediction
        logger.info("Step 8: Cost & Urgency Prediction")
        start = time.time()
        cost_urgency = self.cost_predictor.predict_all(
            features,
            severity_result["score"],
            severity_result["class"]
        )
        timing["cost_prediction"] = time.time() - start
        
        timing["total"] = time.time() - start_total
        
        # Compile results
        results = {
            "status": "damage_detected",
            "input_image": str(image_path),
            "detection": {
                "is_damaged": is_damaged,
                "confidence": float(confidence),
                "probabilities": probs.tolist()
            },
            "measurements": measurement_report,
            "severity": severity_result,
            "cost_urgency": cost_urgency,
            "timing": timing,
            "output_dir": str(output_dir)
        }
        
        # Save outputs
        if save_outputs:
            self._save_outputs(
                results,
                output_dir,
                image_np,
                cleaned_mask,
                edges,
                depth_map_normalized,
                pcd,
                mesh
            )
        
        logger.info(f"Processing complete in {timing['total']:.2f}s")
        logger.info(f"Severity: {severity_result['class'].upper()} (score: {severity_result['score']:.1f})")
        logger.info(f"Estimated cost: ${cost_urgency['cost_prediction']['estimated_cost_usd']:.2f}")
        logger.info(f"Urgency: {cost_urgency['urgency'].upper()}")
        
        return results
    
    def _save_outputs(
        self,
        results: Dict,
        output_dir: Path,
        image: np.ndarray,
        mask: np.ndarray,
        edges: np.ndarray,
        depth_map: np.ndarray,
        pcd,
        mesh
    ):
        """Save all output files."""
        import cv2
        import json
        import open3d as o3d
        
        logger.info(f"Saving outputs to {output_dir}")
        
        # Save JSON report
        with open(output_dir / "report.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._prepare_for_json(results)
            json.dump(json_results, f, indent=2)
        
        # Save mask
        cv2.imwrite(str(output_dir / "mask.png"), mask * 255)
        
        # Save edges
        cv2.imwrite(str(output_dir / "edges.png"), edges)
        
        # Save depth map visualization
        from models.depth.depth_utils import visualize_depth
        depth_vis = visualize_depth(depth_map, colormap="viridis")
        cv2.imwrite(str(output_dir / "depth_map.png"), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
        
        # Save overlay
        overlay = image.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        cv2.imwrite(str(output_dir / "overlay.png"), cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # Save 3D models
        if pcd is not None:
            o3d.io.write_point_cloud(str(output_dir / "point_cloud.ply"), pcd)
        
        if mesh is not None:
            o3d.io.write_triangle_mesh(str(output_dir / "mesh.ply"), mesh)
    
    def _prepare_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def create_pipeline(
    device: str = None,
    pixel_to_mm_ratio: float = 1.0,
    depth_scale: float = 1.0
) -> UDRSPipeline:
    """
    Factory function to create pipeline.
    
    Args:
        device: Device to use
        pixel_to_mm_ratio: 2D calibration
        depth_scale: 3D calibration
        
    Returns:
        UDRSPipeline instance
    """
    return UDRSPipeline(
        device=device,
        pixel_to_mm_ratio=pixel_to_mm_ratio,
        depth_scale=depth_scale
    )
