"""
Synthetic Damage Dataset Generator
Creates realistic synthetic damage images for training
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple, List
from pathlib import Path
import random


class SyntheticDamageGenerator:
    """
    Generate synthetic damage images with ground truth masks.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (640, 480)):
        """
        Initialize generator.
        
        Args:
            image_size: Output image size (width, height)
        """
        self.image_size = image_size
    
    def generate_base_texture(self) -> np.ndarray:
        """
        Generate base surface texture.
        
        Returns:
            RGB image
        """
        # Random concrete/metal/asphalt texture
        base_color = random.choice([
            (180, 180, 180),  # Concrete
            (100, 100, 100),  # Asphalt
            (200, 200, 210),  # Metal
            (150, 120, 100)   # Wood
        ])
        
        img = np.full((self.image_size[1], self.image_size[0], 3), base_color, dtype=np.uint8)
        
        # Add noise
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Add Gaussian blur for texture
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        return img
    
    def generate_crack(
        self,
        length: int = 200,
        width: int = 3,
        branching: bool = True
    ) -> np.ndarray:
        """
        Generate crack pattern.
        
        Args:
            length: Approximate crack length in pixels
            width: Crack width
            branching: Generate branching cracks
            
        Returns:
            Binary mask of crack
        """
        mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
        
        # Start position
        x, y = random.randint(50, self.image_size[0] - 50), random.randint(50, self.image_size[1] - 50)
        
        # Random walk to create crack
        angle = random.uniform(0, 2 * np.pi)
        points = [(x, y)]
        
        for _ in range(length):
            # Random walk with smoothness
            angle += random.uniform(-0.3, 0.3)
            step = random.uniform(1, 3)
            x += int(step * np.cos(angle))
            y += int(step * np.sin(angle))
            
            # Boundary check
            x = np.clip(x, 0, self.image_size[0] - 1)
            y = np.clip(y, 0, self.image_size[1] - 1)
            
            points.append((x, y))
        
        # Draw crack
        for i in range(len(points) - 1):
            cv2.line(mask, points[i], points[i + 1], 255, width)
        
        # Add branches
        if branching and len(points) > 20:
            num_branches = random.randint(1, 3)
            for _ in range(num_branches):
                branch_start = points[random.randint(10, len(points) - 10)]
                branch_mask = self.generate_crack(length//3, width-1, branching=False)
                # Shift to start from branch_start
                mask = np.maximum(mask, branch_mask)
        
        # Smooth edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def generate_dent(
        self,
        radius: int = 50,
        depth: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dent/impact damage.
        
        Args:
            radius: Dent radius
            depth: Relative depth (0-1)
            
        Returns:
            Tuple of (mask, depth_map)
        """
        mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
        depth_map = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float32)
        
        # Random center
        cx = random.randint(radius + 20, self.image_size[0] - radius - 20)
        cy = random.randint(radius + 20, self.image_size[1] - radius - 20)
        
        # Create Gaussian dent
        y, x = np.ogrid[:self.image_size[1], :self.image_size[0]]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Gaussian profile
        sigma = radius / 2
        gaussian = np.exp(-(dist**2) / (2 * sigma**2))
        depth_map = gaussian * depth
        
        # Create binary mask (threshold)
        mask[depth_map > 0.1 * depth] = 255
        
        return mask, depth_map
    
    def generate_corrosion(
        self,
        severity: float = 0.3
    ) -> np.ndarray:
        """
        Generate corrosion pattern.
        
        Args:
            severity: Corrosion severity (0-1)
            
        Returns:
            Binary mask
        """
        mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
        
        # Multiple corrosion spots
        num_spots = int(severity * 20) + 5
        
        for _ in range(num_spots):
            cx = random.randint(0, self.image_size[0])
            cy = random.randint(0, self.image_size[1])
            radius = random.randint(10, 40)
            
            cv2.circle(mask, (cx, cy), radius, 255, -1)
        
        # Add irregular edges using noise
        noise = np.random.randint(0, 100, mask.shape, dtype=np.uint8)
        mask = cv2.bitwise_and(mask, noise)
        
        # Threshold and clean
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def apply_damage_to_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        damage_type: str = "crack",
        depth_map: np.ndarray = None
    ) -> np.ndarray:
        """
        Apply damage visualization to image.
        
        Args:
            image: Base image
            mask: Damage mask
            damage_type: Type of damage
            depth_map: Optional depth map for dents
            
        Returns:
            Image with damage
        """
        result = image.copy()
        
        if damage_type == "crack":
            # Darken crack regions
            result[mask > 0] = result[mask > 0] * 0.3
        
        elif damage_type == "dent" and depth_map is not None:
            # Apply shading based on depth
            shading = (1 - depth_map * 0.7)
            for c in range(3):
                result[:, :, c] = (result[:, :, c] * shading).astype(np.uint8)
        
        elif damage_type == "corrosion":
            # Add rust color
            rust_color = np.array([50, 70, 150], dtype=np.uint8)  # BGR rust
            result[mask > 0] = result[mask > 0] * 0.4 + rust_color * 0.6
        
        return result
    
    def generate_dataset(
        self,
        output_dir: Path,
        num_samples: int = 100,
        damage_types: List[str] = None
    ):
        """
        Generate complete synthetic dataset.
        
        Args:
            output_dir: Output directory
            num_samples: Number of samples to generate
            damage_types: List of damage types (crack, dent, corrosion)
        """
        if damage_types is None:
            damage_types = ["crack", "dent", "corrosion"]
        
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {num_samples} synthetic samples...")
        
        for i in range(num_samples):
            # Generate base
            image = self.generate_base_texture()
            
            # Random damage type
            damage_type = random.choice(damage_types)
            
            # Generate damage
            if damage_type == "crack":
                mask = self.generate_crack(
                    length=random.randint(100, 300),
                    width=random.randint(2, 5),
                    branching=random.choice([True, False])
                )
                depth_map = None
            elif damage_type == "dent":
                mask, depth_map = self.generate_dent(
                    radius=random.randint(30, 80),
                    depth=random.uniform(0.2, 0.5)
                )
            else:  # corrosion
                mask = self.generate_corrosion(severity=random.uniform(0.2, 0.6))
                depth_map = None
            
            # Apply damage
            damaged_image = self.apply_damage_to_image(image, mask, damage_type, depth_map)
            
            # Save
            cv2.imwrite(str(images_dir / f"{damage_type}_{i:04d}.png"), damaged_image)
            cv2.imwrite(str(masks_dir / f"{damage_type}_{i:04d}.png"), mask)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        print(f"Dataset generation complete! Saved to {output_dir}")


def generate_synthetic_dataset(
    output_dir: Path,
    num_samples: int = 100,
    image_size: Tuple[int, int] = (640, 480)
):
    """
    Helper function to generate synthetic dataset.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples
        image_size: Image dimensions
    """
    generator = SyntheticDamageGenerator(image_size=image_size)
    generator.generate_dataset(output_dir, num_samples)


if __name__ == "__main__":
    # Example usage
    output_dir = Path("data/synthetic")
    generate_synthetic_dataset(output_dir, num_samples=100)
