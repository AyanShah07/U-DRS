"""
Severity Scoring Model
Predicts damage severity score (0-100) and classification
"""

import numpy as np
from typing import Tuple, Dict
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


class SeverityScorer:
    """
    Damage severity scoring and classification.
    """
    
    def __init__(
        self,
        model_path: Path = None,
        use_ml: bool = False
    ):
        """
        Initialize severity scorer.
        
        Args:
            model_path: Path to trained ML model (optional)
            use_ml: Use ML model or rule-based scoring
        """
        self.use_ml = use_ml
        self.model = None
        
        if use_ml and model_path and model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        
        # Define severity thresholds
        self.thresholds = {
            "minor": (0, 25),
            "moderate": (25, 50),
            "severe": (50, 75),
            "critical": (75, 100)
        }
    
    def score_rule_based(self, features: np.ndarray) -> float:
        """
        Rule-based severity scoring.
        
        Args:
            features: Feature array from DamageAnalyzer
                [area, perimeter, crack_length, crack_width_mean, crack_width_max,
                 bbox_width, bbox_height, aspect_ratio, compactness, circularity,
                 num_regions, max_depth, mean_depth, std_depth, range_depth,
                 deformation_volume, mean_deformation, max_deformation]
        
        Returns:
            Severity score (0-100)
        """
        score = 0.0
        
        # Extract features
        area = features[0]
        crack_length = features[2]
        crack_width_max = features[4]
        max_depth = features[11] if len(features) > 11 else 0.0
        max_deformation = features[17] if len(features) > 17 else 0.0
        num_regions = features[10]
        
        # Area contribution (0-30 points)
        if area < 100:
            score += 5
        elif area < 500:
            score += 10
        elif area < 1000:
            score += 15
        elif area < 5000:
            score += 20
        else:
            score += 30
        
        # Length contribution (0-20 points)
        if crack_length < 10:
            score += 2
        elif crack_length < 50:
            score += 5
        elif crack_length < 100:
            score += 10
        elif crack_length < 200:
            score += 15
        else:
            score += 20
        
        # Width contribution (0-15 points)
        if crack_width_max < 1:
            score += 2
        elif crack_width_max < 3:
            score += 5
        elif crack_width_max < 5:
            score += 10
        else:
            score += 15
        
        # Depth contribution (0-20 points)
        if max_depth > 0:
            if max_depth < 5:
                score += 5
            elif max_depth < 10:
                score += 10
            elif max_depth < 20:
                score += 15
            else:
                score += 20
        
        # Deformation contribution (0-10 points)
        if max_deformation > 0:
            if max_deformation < 5:
                score += 3
            elif max_deformation < 10:
                score += 7
            else:
                score += 10
        
        # Multiple regions penalty (0-5 points)
        if num_regions > 1:
            score += min(num_regions * 2, 5)
        
        return float(np.clip(score, 0, 100))
    
    def score_ml_based(self, features: np.ndarray) -> float:
        """
        ML-based severity scoring.
        
        Args:
            features: Feature array
            
        Returns:
            Severity score (0-100)
        """
        if self.model is None:
            # Fallback to rule-based
            return self.score_rule_based(features)
        
        # Predict severity class probabilities
        features_reshaped = features.reshape(1, -1)
        proba = self.model.predict_proba(features_reshaped)[0]
        
        # Convert to score: weighted average of class midpoints
        class_scores = [12.5, 37.5, 62.5, 87.5]  # Midpoints of severity ranges
        score = np.dot(proba, class_scores)
        
        return float(score)
    
    def score(self, features: np.ndarray) -> float:
        """
        Calculate severity score.
        
        Args:
            features: Feature array
            
        Returns:
            Severity score (0-100)
        """
        if self.use_ml:
            return self.score_ml_based(features)
        else:
            return self.score_rule_based(features)
    
    def classify(self, score: float) -> str:
        """
        Classify severity level from score.
        
        Args:
            score: Severity score
            
        Returns:
            Severity class ('minor', 'moderate', 'severe', 'critical')
        """
        for level, (low, high) in self.thresholds.items():
            if low <= score < high:
                return level
        
        return "critical"  # Default for score >= 75
    
    def predict(self, features: np.ndarray) -> Dict[str, any]:
        """
        Predict severity score and classification.
        
        Args:
            features: Feature array
            
        Returns:
            Dictionary with score, class, and confidence
        """
        score = self.score(features)
        severity_class = self.classify(score)
        
        # Confidence (distance from class boundaries)
        low, high = self.thresholds[severity_class]
        mid = (low + high) / 2
        confidence = 1.0 - abs(score - mid) / ((high - low) / 2)
        
        return {
            "score": float(score),
            "class": severity_class,
            "confidence": float(confidence),
            "threshold_low": float(low),
            "threshold_high": float(high)
        }


def create_severity_scorer(
    model_path: Path = None,
    use_ml: bool = False
) -> SeverityScorer:
    """
    Factory function to create severity scorer.
    
    Args:
        model_path: Path to trained model
        use_ml: Use ML or rule-based
        
    Returns:
        SeverityScorer instance
    """
    return SeverityScorer(model_path=model_path, use_ml=use_ml)
