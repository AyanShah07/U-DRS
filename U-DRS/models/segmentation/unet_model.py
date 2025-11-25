"""
U-Net Segmentation Module
Standard U-Net architecture for damage segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class UNetBlock(nn.Module):
    """Basic U-Net convolutional block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for semantic segmentation.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list = [64, 128, 256, 512]):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary segmentation)
            features: Feature dimensions for each level
        """
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(UNetBlock(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(UNetBlock(feature * 2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)


class DamageSegmenter:
    """
    Inference wrapper for damage segmentation.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5,
        input_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize segmentation engine.
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on
            threshold: Segmentation threshold for binary mask
            input_size: Input image size (width, height)
        """
        self.device = device
        self.threshold = threshold
        self.input_size = input_size
        
        # Initialize model
        self.model = UNet(in_channels=3, out_channels=1)
        
        # Load weights if provided
        if model_path and model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def segment(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment damage in image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (binary_mask, probability_map)
                - binary_mask: Binary segmentation mask (0/1)
                - probability_map: Probability scores (0-1)
        """
        original_size = image.size  # (width, height)
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        output = self.model(img_tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Resize to original size
        prob_map_pil = Image.fromarray((prob_map * 255).astype(np.uint8))
        prob_map_resized = prob_map_pil.resize(original_size, Image.BILINEAR)
        prob_map = np.array(prob_map_resized) / 255.0
        
        # Apply threshold
        binary_mask = (prob_map >= self.threshold).astype(np.uint8)
        
        return binary_mask, prob_map
    
    def segment_from_path(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment damage from image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (binary_mask, probability_map)
        """
        image = Image.open(image_path).convert("RGB")
        return self.segment(image)
    
    def get_damage_area(self, binary_mask: np.ndarray, pixel_to_mm_ratio: float = 1.0) -> float:
        """
        Calculate damaged area in mm².
        
        Args:
            binary_mask: Binary segmentation mask
            pixel_to_mm_ratio: Conversion ratio from pixels to mm
            
        Returns:
            Area in mm²
        """
        num_damaged_pixels = np.sum(binary_mask)
        area_mm2 = num_damaged_pixels * (pixel_to_mm_ratio ** 2)
        return area_mm2
    
    def get_bounding_box(self, binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of damaged region.
        
        Args:
            binary_mask: Binary segmentation mask
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) or None if no damage
        """
        coords = np.column_stack(np.where(binary_mask > 0))
        
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return int(x_min), int(y_min), int(x_max), int(y_max)


def create_segmenter(
    model_path: Optional[Path] = None,
    device: str = "cuda",
    input_size: Tuple[int, int] = (512, 512)
) -> DamageSegmenter:
    """
    Factory function to create segmenter instance.
    
    Args:
        model_path: Path to trained model
        device: Device to use
        input_size: Input image size
        
    Returns:
        DamageSegmenter instance
    """
    return DamageSegmenter(model_path=model_path, device=device, input_size=input_size)
