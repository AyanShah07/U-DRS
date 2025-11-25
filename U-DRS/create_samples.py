"""
Quick Sample Generator
Creates sample damage images for testing U-DRS
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset_generator import SyntheticDamageGenerator
from PIL import Image
import cv2
import numpy as np


def create_sample_images(output_dir: Path, num_samples: int = 5):
    """
    Create a few sample damage images for quick testing.
    
    Args:
        output_dir: Directory to save samples
        num_samples: Number of samples per damage type
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticDamageGenerator(image_size=(640, 480))
    
    print(f"Generating {num_samples} sample images of each type...")
    
    # Generate cracks
    print("\n✓ Generating crack samples...")
    for i in range(num_samples):
        image = generator.generate_base_texture()
        mask = generator.generate_crack(
            length=150 + i * 30,
            width=2 + i,
            branching=(i % 2 == 0)
        )
        damaged_image = generator.apply_damage_to_image(image, mask, "crack")
        
        # Save
        cv2.imwrite(str(output_dir / f"crack_{i+1}.jpg"), damaged_image)
        cv2.imwrite(str(output_dir / f"crack_{i+1}_mask.png"), mask)
    
    # Generate dents
    print("✓ Generating dent samples...")
    for i in range(num_samples):
        image = generator.generate_base_texture()
        mask, depth_map = generator.generate_dent(
            radius=30 + i * 10,
            depth=0.2 + i * 0.1
        )
        damaged_image = generator.apply_damage_to_image(image, mask, "dent", depth_map)
        
        cv2.imwrite(str(output_dir / f"dent_{i+1}.jpg"), damaged_image)
        cv2.imwrite(str(output_dir / f"dent_{i+1}_mask.png"), mask)
    
    # Generate corrosion
    print("✓ Generating corrosion samples...")
    for i in range(num_samples):
        image = generator.generate_base_texture()
        mask = generator.generate_corrosion(severity=0.2 + i * 0.1)
        damaged_image = generator.apply_damage_to_image(image, mask, "corrosion")
        
        cv2.imwrite(str(output_dir / f"corrosion_{i+1}.jpg"), damaged_image)
        cv2.imwrite(str(output_dir / f"corrosion_{i+1}_mask.png"), mask)
    
    # Generate one intact sample
    print("✓ Generating intact sample...")
    intact_image = generator.generate_base_texture()
    cv2.imwrite(str(output_dir / "intact.jpg"), intact_image)
    
    print(f"\n{'='*60}")
    print(f"✅ Generated {num_samples * 3 + 1} sample images!")
    print(f"Location: {output_dir}")
    print(f"{'='*60}\n")
    print("Sample images:")
    print("  - crack_1.jpg to crack_5.jpg (with masks)")
    print("  - dent_1.jpg to dent_5.jpg (with masks)")
    print("  - corrosion_1.jpg to corrosion_5.jpg (with masks)")
    print("  - intact.jpg (no damage)")
    print("\nTest with:")
    print(f"  python run_inference.py --input {output_dir}/crack_1.jpg")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample damage images")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/samples",
        help="Output directory (default: data/samples)"
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=5,
        help="Number of samples per type (default: 5)"
    )
    
    args = parser.parse_args()
    
    create_sample_images(Path(args.output), args.num_samples)
