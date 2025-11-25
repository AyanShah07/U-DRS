"""
Damage Detection Module
Uses ResNet18-based CNN for binary classification (Damaged vs Intact)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class DamageDetector(nn.Module):
    """
    ResNet18-based damage detector with custom classification head.
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        """
        Initialize damage detector.
        
        Args:
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of output classes (default: 2 for binary)
        """
        super(DamageDetector, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability scores."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


class DamageDetectorInference:
    """
    Inference wrapper for damage detection.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on
            threshold: Detection confidence threshold
        """
        self.device = device
        self.threshold = threshold
        
        # Initialize model
        self.model = DamageDetector(pretrained=True)
        
        # Load weights if provided
        if model_path and model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def detect(self, image: Image.Image) -> Tuple[bool, float, np.ndarray]:
        """
        Detect damage in image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (is_damaged, confidence, class_probabilities)
        """
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        probs = self.model.predict_proba(img_tensor)
        probs_np = probs.cpu().numpy()[0]
        
        # Get prediction
        damage_prob = probs_np[1]  # Probability of damage class
        is_damaged = damage_prob >= self.threshold
        
        return is_damaged, float(damage_prob), probs_np
    
    def detect_from_path(self, image_path: Path) -> Tuple[bool, float, np.ndarray]:
        """
        Detect damage from image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_damaged, confidence, class_probabilities)
        """
        image = Image.open(image_path).convert("RGB")
        return self.detect(image)
    
    def get_activation_map(self, image: Image.Image, layer_name: str = "layer4") -> np.ndarray:
        """
        Generate Grad-CAM activation map for visualization.
        
        Args:
            image: PIL Image
            layer_name: Name of layer to extract activations from
            
        Returns:
            Activation map as numpy array
        """
        # This is a simplified version - full Grad-CAM implementation would be more complex
        self.model.eval()
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get activations from target layer
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach())
        
        # Register hook
        target_layer = dict(self.model.backbone.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)
        
        # Forward pass
        output = self.model(img_tensor)
        
        # Remove hook
        handle.remove()
        
        # Get class activation map
        if activations:
            activation = activations[0].cpu().numpy()[0]
            # Average across channels and resize
            cam = np.mean(activation, axis=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam
        
        return np.zeros((7, 7))  # Default empty CAM


def create_detector(model_path: Optional[Path] = None, device: str = "cuda") -> DamageDetectorInference:
    """
    Factory function to create detector instance.
    
    Args:
        model_path: Path to trained model
        device: Device to use
        
    Returns:
        DamageDetectorInference instance
    """
    return DamageDetectorInference(model_path=model_path, device=device)
