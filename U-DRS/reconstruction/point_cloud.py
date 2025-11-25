"""
3D Point Cloud Generation
Converts depth map + RGB image to 3D point cloud
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple
from PIL import Image


class PointCloudGenerator:
    """
    Generates 3D point clouds from depth maps and images.
    """
    
    def __init__(
        self,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        fov: float = 60.0
    ):
        """
        Initialize point cloud generator.
        
        Args:
            fx, fy: Camera focal lengths (pixels)
            cx, cy: Camera principal point (pixels)
            fov: Field of view in degrees (used if fx/fy not provided)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.fov = fov
    
    def set_camera_intrinsics(
        self,
        width: int,
        height: int,
        fx: Optional[float] = None,
        fy: Optional[float] = None
    ):
        """
        Set or estimate camera intrinsics.
        
        Args:
            width: Image width
            height: Image height
            fx, fy: Optional focal lengths
        """
        if fx is None or fy is None:
            # Estimate from FOV
            fov_rad = np.deg2rad(self.fov)
            self.fx = width / (2 * np.tan(fov_rad / 2))
            self.fy = self.fx
        else:
            self.fx = fx
            self.fy = fy
        
        # Principal point at center
        if self.cx is None:
            self.cx = width / 2
        if self.cy is None:
            self.cy = height / 2
    
    def depth_to_pointcloud(
        self,
        depth_map: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        depth_scale: float = 1.0
    ) -> o3d.geometry.PointCloud:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            depth_map: Depth map (H x W)
            rgb_image: Optional RGB image for coloring
            mask: Optional binary mask (only generate points in masked region)
            depth_scale: Scale factor for depth values
            
        Returns:
            Open3D PointCloud object
        """
        h, w = depth_map.shape
        
        # Set camera intrinsics if not already set
        if self.fx is None or self.fy is None:
            self.set_camera_intrinsics(w, h)
        
        # Create meshgrid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply mask if provided
        if mask is not None:
            valid = mask > 0
        else:
            valid = np.ones((h, w), dtype=bool)
        
        # Filter out invalid depths
        valid = valid & (depth_map > 0)
        
        # Convert to 3D coordinates
        z = depth_map[valid] * depth_scale
        x = (u[valid] - self.cx) * z / self.fx
        y = (v[valid] - self.cy) * z / self.fy
        
        # Stack into (N, 3) array
        points = np.stack([x, y, z], axis=-1)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors if RGB image provided
        if rgb_image is not None:
            if rgb_image.shape[:2] != (h, w):
                rgb_image = Image.fromarray(rgb_image).resize((w, h))
                rgb_image = np.array(rgb_image)
            
            colors = rgb_image[valid].astype(np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def filter_pointcloud(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
        voxel_size: Optional[float] = None
    ) -> o3d.geometry.PointCloud:
        """
        Filter and clean point cloud.
        
        Args:
            pcd: Input point cloud
            nb_neighbors: Number of neighbors for statistical outlier removal
            std_ratio: Standard deviation ratio for outlier removal
            voxel_size: Voxel size for downsampling (None to skip)
            
        Returns:
            Filtered point cloud
        """
        # Statistical outlier removal
        pcd_filtered, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        # Voxel downsampling
        if voxel_size is not None:
            pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size=voxel_size)
        
        return pcd_filtered
    
    def estimate_normals(
        self,
        pcd: o3d.geometry.PointCloud,
        search_param: Optional[o3d.geometry.KDTreeSearchParam] = None
    ) -> o3d.geometry.PointCloud:
        """
        Estimate point cloud normals.
        
        Args:
            pcd: Point cloud
            search_param: Search parameters for normal estimation
            
        Returns:
            Point cloud with normals
        """
        if search_param is None:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        
        pcd.estimate_normals(search_param=search_param)
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        return pcd
    
    def get_bounding_box(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box of point cloud.
        
        Args:
            pcd: Point cloud
            
        Returns:
            Tuple of (min_bound, max_bound)
        """
        aabb = pcd.get_axis_aligned_bounding_box()
        return np.array(aabb.min_bound), np.array(aabb.max_bound)
    
    def compute_volume(self, pcd: o3d.geometry.PointCloud) -> float:
        """
        Estimate volume of point cloud (using convex hull).
        
        Args:
            pcd: Point cloud
            
        Returns:
            Volume estimate
        """
        try:
            hull, _ = pcd.compute_convex_hull()
            volume = hull.get_volume()
            return volume
        except:
            return 0.0


def create_pointcloud_generator(fov: float = 60.0) -> PointCloudGenerator:
    """
    Factory function to create point cloud generator.
    
    Args:
        fov: Field of view in degrees
        
    Returns:
        PointCloudGenerator instance
    """
    return PointCloudGenerator(fov=fov)
