"""
Mesh Reconstruction
Converts point cloud to mesh using Poisson or Ball-Pivoting algorithms
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple
from pathlib import Path


class MeshBuilder:
    """
    Builds mesh from point cloud.
    """
    
    def __init__(self, method: str = "poisson"):
        """
        Initialize mesh builder.
        
        Args:
            method: Reconstruction method (poisson or ball_pivoting)
        """
        self.method = method
    
    def build_poisson(
        self,
        pcd: o3d.geometry.PointCloud,
        depth: int = 9,
        width: int = 0,
        scale: float = 1.1,
        linear_fit: bool = False
    ) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """
        Build mesh using Poisson surface reconstruction.
        
        Args:
            pcd: Point cloud with normals
            depth: Octree depth (higher = more detail, slower)
            width: Target width of grid
            scale: Surface bounding box scale
            linear_fit: Use linear interpolation
            
        Returns:
            Tuple of (mesh, densities)
        """
        # Ensure normals are estimated
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit
        )
        
        return mesh, densities
    
    def build_ball_pivoting(
        self,
        pcd: o3d.geometry.PointCloud,
        radii: Optional[list] = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Build mesh using Ball-Pivoting algorithm.
        
        Args:
            pcd: Point cloud with normals
            radii: List of radii for ball pivoting
            
        Returns:
            Triangle mesh
        """
        # Ensure normals are estimated
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        # Default radii based on point cloud scale
        if radii is None:
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist * 1.5, avg_dist * 3]
        
        # Ball pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )
        
        return mesh
    
    def build(
        self,
        pcd: o3d.geometry.PointCloud,
        **kwargs
    ) -> o3d.geometry.TriangleMesh:
        """
        Build mesh using configured method.
        
        Args:
            pcd: Point cloud
            **kwargs: Method-specific arguments
            
        Returns:
            Triangle mesh
        """
        if self.method == "poisson":
            mesh, densities = self.build_poisson(pcd, **kwargs)
            
            # Optional: remove low-density vertices
            if "density_threshold" in kwargs:
                vertices_to_remove = densities < np.quantile(densities, kwargs["density_threshold"])
                mesh.remove_vertices_by_mask(vertices_to_remove)
            
            return mesh
        
        elif self.method == "ball_pivoting":
            return self.build_ball_pivoting(pcd, **kwargs)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def refine_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        iterations: int = 1
    ) -> o3d.geometry.TriangleMesh:
        """
        Refine mesh using smoothing and subdivision.
        
        Args:
            mesh: Input mesh
            iterations: Number of smoothing iterations
            
        Returns:
            Refined mesh
        """
        # Laplacian smoothing
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
        
        # Compute normals
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        return mesh
    
    def simplify_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_triangles: int = 10000
    ) -> o3d.geometry.TriangleMesh:
        """
        Simplify mesh by reducing triangle count.
        
        Args:
            mesh: Input mesh
            target_triangles: Target number of triangles
            
        Returns:
            Simplified mesh
        """
        mesh_simplified = mesh.simplify_quadric_decimation(target_triangles)
        mesh_simplified.compute_vertex_normals()
        
        return mesh_simplified
    
    def save_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        path: Path,
        write_vertex_colors: bool = True,
        write_vertex_normals: bool = True
    ):
        """
        Save mesh to file.
        
        Args:
            mesh: Mesh to save
            path: Output file path (.ply, .obj, .stl)
            write_vertex_colors: Include vertex colors
            write_vertex_normals: Include vertex normals
        """
        o3d.io.write_triangle_mesh(
            str(path),
            mesh,
            write_vertex_colors=write_vertex_colors,
            write_vertex_normals=write_vertex_normals
        )
    
    def get_mesh_info(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        """
        Get mesh statistics.
        
        Args:
            mesh: Triangle mesh
            
        Returns:
            Dictionary of mesh info
        """
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        info = {
            "num_vertices": len(vertices),
            "num_triangles": len(triangles),
            "is_watertight": mesh.is_watertight(),
            "is_orientable": mesh.is_orientable(),
            "surface_area": mesh.get_surface_area(),
            "volume": mesh.get_volume() if mesh.is_watertight() else 0.0
        }
        
        # Bounding box
        bbox = mesh.get_axis_aligned_bounding_box()
        info["bbox_min"] = np.array(bbox.min_bound)
        info["bbox_max"] = np.array(bbox.max_bound)
        info["bbox_dimensions"] = info["bbox_max"] - info["bbox_min"]
        
        return info


def create_mesh_builder(method: str = "poisson") -> MeshBuilder:
    """
    Factory function to create mesh builder.
    
    Args:
        method: Reconstruction method
        
    Returns:
        MeshBuilder instance
    """
    return MeshBuilder(method=method)
