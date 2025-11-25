"""
Evaluation Metrics for U-DRS System
Includes 2D segmentation, 3D reconstruction, and depth estimation metrics
"""

import numpy as np
from typing import Dict, Tuple
import open3d as o3d
from sklearn.metrics import precision_score, recall_score, f1_score


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for damage reconstruction.
    """
    
    @staticmethod
    def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Dice coefficient for segmentation.
        
        Args:
            pred: Predicted mask
            gt: Ground truth mask
            
        Returns:
            Dice score (0-1)
        """
        pred_binary = (pred > 0).astype(np.uint8)
        gt_binary = (gt > 0).astype(np.uint8)
        
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = (2.0 * intersection) / union
        return float(dice)
    
    @staticmethod
    def iou(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Intersection over Union.
        
        Args:
            pred: Predicted mask
            gt: Ground truth mask
            
        Returns:
            IoU score (0-1)
        """
        pred_binary = (pred > 0).astype(np.uint8)
        gt_binary = (gt > 0).astype(np.uint8)
        
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou_score = intersection / union
        return float(iou_score)
    
    @staticmethod
    def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate pixel-wise accuracy.
        
        Args:
            pred: Predicted mask
            gt: Ground truth mask
            
        Returns:
            Accuracy (0-1)
        """
        pred_binary = (pred > 0).astype(np.uint8)
        gt_binary = (gt > 0).astype(np.uint8)
        
        correct = np.sum(pred_binary == gt_binary)
        total = pred_binary.size
        
        return float(correct / total)
    
    @staticmethod
    def precision_recall_f1(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            pred: Predicted mask
            gt: Ground truth mask
            
        Returns:
            Dictionary with precision, recall, F1
        """
        pred_flat = (pred > 0).astype(np.uint8).flatten()
        gt_flat = (gt > 0).astype(np.uint8).flatten()
        
        precision = precision_score(gt_flat, pred_flat, zero_division=0)
        recall = recall_score(gt_flat, pred_flat, zero_division=0)
        f1 = f1_score(gt_flat, pred_flat, zero_division=0)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    @staticmethod
    def chamfer_distance(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> float:
        """
        Calculate Chamfer distance between two point clouds.
        
        Args:
            pcd1: First point cloud
            pcd2: Second point cloud
            
        Returns:
            Average Chamfer distance
        """
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        
        # Build KD-tree for pcd2
        pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
        
        # Distance from pcd1 to pcd2
        distances_1to2 = []
        for point in points1:
            [_, idx, dist] = pcd2_tree.search_knn_vector_3d(point, 1)
            distances_1to2.append(np.sqrt(dist[0]))
        
        # Build KD-tree for pcd1
        pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
        
        # Distance from pcd2 to pcd1
        distances_2to1 = []
        for point in points2:
            [_, idx, dist] = pcd1_tree.search_knn_vector_3d(point, 1)
            distances_2to1.append(np.sqrt(dist[0]))
        
        # Average Chamfer distance
        chamfer_dist = (np.mean(distances_1to2) + np.mean(distances_2to1)) / 2
        return float(chamfer_dist)
    
    @staticmethod
    def hausdorff_distance(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> float:
        """
        Calculate Hausdorff distance between point clouds.
        
        Args:
            pcd1: First point cloud
            pcd2: Second point cloud
            
        Returns:
            Hausdorff distance
        """
        from scipy.spatial.distance import directed_hausdorff
        
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        
        dist_1to2 = directed_hausdorff(points1, points2)[0]
        dist_2to1 = directed_hausdorff(points2, points1)[0]
        
        hausdorff_dist = max(dist_1to2, dist_2to1)
        return float(hausdorff_dist)
    
    @staticmethod
    def depth_rmse(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Calculate RMSE for depth prediction.
        
        Args:
            pred_depth: Predicted depth map
            gt_depth: Ground truth depth map
            mask: Optional mask for valid regions
            
        Returns:
            RMSE
        """
        if mask is not None:
            pred = pred_depth[mask > 0]
            gt = gt_depth[mask > 0]
        else:
            pred = pred_depth.flatten()
            gt = gt_depth.flatten()
        
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        return float(rmse)
    
    @staticmethod
    def depth_mae(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Calculate MAE for depth prediction.
        
        Args:
            pred_depth: Predicted depth map
            gt_depth: Ground truth depth map
            mask: Optional mask
            
        Returns:
            MAE
        """
        if mask is not None:
            pred = pred_depth[mask > 0]
            gt = gt_depth[mask > 0]
        else:
            pred = pred_depth.flatten()
            gt = gt_depth.flatten()
        
        mae = np.mean(np.abs(pred - gt))
        return float(mae)
    
    @staticmethod
    def delta_accuracy(pred_depth: np.ndarray, gt_depth: np.ndarray, threshold: float = 1.25) -> float:
        """
        Calculate delta accuracy (percentage of pixels within threshold).
        
        Args:
            pred_depth: Predicted depth
            gt_depth: Ground truth depth
            threshold: Delta threshold
            
        Returns:
            Accuracy percentage
        """
        ratio = np.maximum(pred_depth / (gt_depth + 1e-8), gt_depth / (pred_depth + 1e-8))
        accuracy = np.mean(ratio < threshold)
        return float(accuracy)
    
    @staticmethod
    def evaluate_segmentation(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Complete segmentation evaluation.
        
        Args:
            pred: Predicted mask
            gt: Ground truth mask
            
        Returns:
            Dictionary of all metrics
        """
        metrics = EvaluationMetrics()
        
        return {
            "dice": metrics.dice_coefficient(pred, gt),
            "iou": metrics.iou(pred, gt),
            "pixel_accuracy": metrics.pixel_accuracy(pred, gt),
            **metrics.precision_recall_f1(pred, gt)
        }
    
    @staticmethod
    def evaluate_depth(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
        """
        Complete depth evaluation.
        
        Args:
            pred: Predicted depth
            gt: Ground truth depth
            mask: Optional mask
            
        Returns:
            Dictionary of metrics
        """
        metrics = EvaluationMetrics()
        
        return {
            "rmse": metrics.depth_rmse(pred, gt, mask),
            "mae": metrics.depth_mae(pred, gt, mask),
            "delta_1.25": metrics.delta_accuracy(pred, gt, 1.25),
            "delta_1.25^2": metrics.delta_accuracy(pred, gt, 1.25**2),
            "delta_1.25^3": metrics.delta_accuracy(pred, gt, 1.25**3)
        }
    
    @staticmethod
    def evaluate_3d_reconstruction(
        pred_pcd: o3d.geometry.PointCloud,
        gt_pcd: o3d.geometry.PointCloud
    ) -> Dict[str, float]:
        """
        Complete 3D reconstruction evaluation.
        
        Args:
            pred_pcd: Predicted point cloud
            gt_pcd: Ground truth point cloud
            
        Returns:
            Dictionary of metrics
        """
        metrics = EvaluationMetrics()
        
        return {
            "chamfer_distance": metrics.chamfer_distance(pred_pcd, gt_pcd),
            "hausdorff_distance": metrics.hausdorff_distance(pred_pcd, gt_pcd)
        }
