"""
3D Visualization Tools
Interactive visualization of point clouds and meshes
"""

import open3d as o3d
import numpy as np
from pathlib import Path
from typing import List, Optional, Union


class Visualizer3D:
    """
    3D visualization for point clouds and meshes.
    """
    
    def __init__(self, window_name: str = "U-DRS 3D Viewer"):
        """
        Initialize visualizer.
        
        Args:
            window_name: Window title
        """
        self.window_name = window_name
    
    def visualize(
        self,
        geometries: Union[o3d.geometry.Geometry, List[o3d.geometry.Geometry]],
        show_coordinate_frame: bool = True,
        point_size: float = 1.0
    ):
        """
        Visualize geometries interactively.
        
        Args:
            geometries: Single geometry or list of geometries
            show_coordinate_frame: Show XYZ coordinate frame
            point_size: Point size for point clouds
        """
        if not isinstance(geometries, list):
            geometries = [geometries]
        
        # Add coordinate frame
        if show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            geometries.append(coord_frame)
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name)
        
        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Set point size for point clouds
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        
        # Run visualizer
        vis.run()
        vis.destroy_window()
    
    def capture_screenshot(
        self,
        geometries: Union[o3d.geometry.Geometry, List[o3d.geometry.Geometry]],
        output_path: Path,
        width: int = 1920,
        height: int = 1080
    ):
        """
        Capture screenshot of geometries.
        
        Args:
            geometries: Geometries to visualize
            output_path: Output image path
            width: Image width
            height: Image height
        """
        if not isinstance(geometries, list):
            geometries = [geometries]
        
        # Create visualizer (offscreen)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        
        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Update and capture
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(output_path))
        vis.destroy_window()
    
    def create_multiview_screenshots(
        self,
        geometry: o3d.geometry.Geometry,
        output_dir: Path,
        num_views: int = 4
    ) -> List[Path]:
        """
        Create screenshots from multiple viewpoints.
        
        Args:
            geometry: Geometry to visualize
            output_dir: Output directory
            num_views: Number of views around object
            
        Returns:
            List of screenshot paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_paths = []
        
        # Calculate viewpoints around object
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        
        for i, angle in enumerate(angles):
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(geometry)
            
            # Set viewpoint
            ctr = vis.get_view_control()
            ctr.rotate(angle * 180 / np.pi, 0)
            
            # Capture
            output_path = output_dir / f"view_{i:02d}.png"
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(output_path))
            vis.destroy_window()
            
            screenshot_paths.append(output_path)
        
        return screenshot_paths
    
    def visualize_side_by_side(
        self,
        pcd: o3d.geometry.PointCloud,
        mesh: o3d.geometry.TriangleMesh
    ):
        """
        Visualize point cloud and mesh side by side.
        
        Args:
            pcd: Point cloud
            mesh: Mesh
        """
        # Translate mesh for side-by-side view
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        mesh_translated = mesh.translate([extent[0] * 1.5, 0, 0])
        
        # Visualize together
        self.visualize([pcd, mesh_translated])


def create_visualizer(window_name: str = "U-DRS 3D Viewer") -> Visualizer3D:
    """
    Factory function to create visualizer.
    
    Args:
        window_name: Window title
        
    Returns:
        Visualizer3D instance
    """
    return Visualizer3D(window_name=window_name)


def quick_visualize(geometry: o3d.geometry.Geometry, window_name: str = "Quick View"):
    """
    Quick visualization helper.
    
    Args:
        geometry: Geometry to visualize
        window_name: Window title
    """
    o3d.visualization.draw_geometries([geometry], window_name=window_name)
