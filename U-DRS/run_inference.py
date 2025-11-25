"""
Main Entry Point for U-DRS System
Command-line interface for running inference
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pipeline.inference import create_pipeline
from pipeline.config import config
from utils.logger import setup_logger
import json


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="U-DRS: Universal Damage Reconstruction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_inference.py --input crack.jpg

  # With calibration
  python run_inference.py --input dent.jpg --pixel-mm-ratio 0.5 --depth-scale 2.0
  
  # Skip 3D reconstruction
  python run_inference.py --input corrosion.jpg --no-3d
  
  # Specify output directory
  python run_inference.py --input damage.jpg --output results/analysis_001
"""
    )
    
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: data/outputs/<filename>)"
    )
    
    parser.add_argument(
        "--pixel-mm-ratio",
        type=float,
        default=1.0,
        help="Calibration ratio: pixels to mm (default: 1.0)"
    )
    
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Depth scale factor (default: 1.0)"
    )
    
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="Skip 3D reconstruction (faster)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to run on (default: auto-detect)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(config.LOG_FILE, log_level)
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = config.OUTPUTS_DIR / input_path.stem
    
    # Create pipeline
    logger.info("Initializing U-DRS pipeline...")
    device = args.device if args.device else config.DEVICE
    
    pipeline = create_pipeline(
        device=device,
        pixel_to_mm_ratio=args.pixel_mm_ratio,
        depth_scale=args.depth_scale
    )
    
    # Process image
    logger.info(f"Processing: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        results = pipeline.process(
            image_path=input_path,
            save_outputs=True,
            output_dir=output_dir,
            generate_3d=not args.no_3d
        )
        
        # Print summary
        print("\n" + "="*60)
        print("DAMAGE ANALYSIS RESULTS")
        print("="*60)
        
        if results["status"] == "no_damage":
            print(f"\nNo damage detected (confidence: {results['confidence']:.1%})")
        else:
            print(f"\n✓ Damage detected (confidence: {results['detection']['confidence']:.1%})")
            
            # Measurements summary
            summary = results["measurements"]["summary"]
            print(f"\n2D MEASUREMENTS:")
            print(f"  • Damage area: {summary['damage_area_mm2']:.1f} mm²")
            print(f"  • Crack length: {summary['crack_length_mm']:.1f} mm")
            print(f"  • Crack width (mean): {summary['crack_width_mean_mm']:.2f} mm")
            print(f"  • Crack width (max): {summary['crack_width_max_mm']:.2f} mm")
            print(f"  • Bounding box: {summary['bbox_width_mm']:.1f} × {summary['bbox_height_mm']:.1f} mm")
            
            # 3D measurements if available
            if "max_depth_mm" in summary and summary["max_depth_mm"]:
                print(f"\n3D MEASUREMENTS:")
                print(f"  • Max depth: {summary['max_depth_mm']:.2f} mm")
                print(f"  • Mean depth: {summary['mean_depth_mm']:.2f} mm")
                if "volume_mm3" in summary:
                    print(f"  • Volume: {summary['volume_mm3']:.1f} mm³")
                if "max_deformation_mm" in summary:
                    print(f"  • Max deformation: {summary['max_deformation_mm']:.2f} mm")
            
            # Severity
            severity = results["severity"]
            print(f"\nSEVERITY ASSESSMENT:")
            print(f"  • Class: {severity['class'].upper()}")
            print(f"  • Score: {severity['score']:.1f} / 100")
            print(f"  • Confidence: {severity['confidence']:.1%}")
            
            # Cost & Urgency
            cost_urgency = results["cost_urgency"]
            cost = cost_urgency["cost_prediction"]
            print(f"\nCOST ESTIMATION:")
            print(f"  • Est. cost: ${cost['estimated_cost_usd']:.2f}")
            print(f"  • Range: ${cost['lower_bound_usd']:.2f} - ${cost['upper_bound_usd']:.2f}")
            
            print(f"\nREPAIR URGENCY:")
            print(f"  • Level: {cost_urgency['urgency'].upper()}")
            print(f"  • Timeline: {cost_urgency['recommended_timeline']}")
            print(f"  • {cost_urgency['urgency_description']}")
            
            # Timing
            timing = results["timing"]
            print(f"\nPROCESSING TIME:")
            print(f"  • Detection: {timing['detection']:.2f}s")
            print(f"  • Segmentation: {timing['segmentation']:.2f}s")
            print(f"  • Depth: {timing['depth_estimation']:.2f}s")
            if "3d_reconstruction" in timing and timing["3d_reconstruction"]:
                print(f"  • 3D Reconstruction: {timing['3d_reconstruction']:.2f}s")
            print(f"  • Total: {timing['total']:.2f}s")
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_dir}")
        print(f"  • Full report: {output_dir / 'report.json'}")
        print(f"  • Mask: {output_dir / 'mask.png'}")
        print(f"  • Overlay: {output_dir / 'overlay.png'}")
        print(f"  • Depth map: {output_dir / 'depth_map.png'}")
        if not args.no_3d:
            print(f"  • Point cloud: {output_dir / 'point_cloud.ply'}")
            print(f"  • Mesh: {output_dir / 'mesh.ply'}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
