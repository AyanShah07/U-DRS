"""
Cost and Urgency Prediction Model
Predicts repair cost and urgency level based on damage measurements
"""

import numpy as np
from typing import Dict, Tuple
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


class CostPredictor:
    """
    Repair cost and urgency prediction.
    """
    
    def __init__(
        self,
        cost_model_path: Path = None,
        base_cost: float = 100.0,
        cost_per_mm2: float = 0.5,
        cost_per_mm_depth: float = 2.0
    ):
        """
        Initialize cost predictor.
        
        Args:
            cost_model_path: Path to trained cost model (optional)
            base_cost: Base repair cost (USD)
            cost_per_mm2: Cost per mm² of damaged area
            cost_per_mm_depth: Cost per mm of depth
        """
        self.base_cost = base_cost
        self.cost_per_mm2 = cost_per_mm2
        self.cost_per_mm_depth = cost_per_mm_depth
        
        self.cost_model = None
        if cost_model_path and cost_model_path.exists():
            with open(cost_model_path, 'rb') as f:
                self.cost_model = pickle.load(f)
    
    def predict_cost_rule_based(
        self,
        features: np.ndarray,
        severity_score: float
    ) -> float:
        """
        Rule-based cost prediction.
        
        Args:
            features: Feature array
            severity_score: Severity score (0-100)
            
        Returns:
            Estimated cost in USD
        """
        # Extract key features
        area = features[0]
        crack_length = features[2]
        max_depth = features[11] if len(features) > 11 else 0.0
        
        # Base cost
        cost = self.base_cost
        
        # Area-based cost
        cost += area * self.cost_per_mm2
        
        # Length-based cost (for cracks)
        cost += crack_length * 0.5
        
        # Depth-based cost
        if max_depth > 0:
            cost += max_depth * self.cost_per_mm_depth
        
        # Severity multiplier
        severity_multiplier = 1.0 + (severity_score / 100.0)
        cost *= severity_multiplier
        
        return float(cost)
    
    def predict_cost_ml_based(
        self,
        features: np.ndarray,
        severity_score: float
    ) -> float:
        """
        ML-based cost prediction.
        
        Args:
            features: Feature array
            severity_score: Severity score
            
        Returns:
            Estimated cost
        """
        if self.cost_model is None:
            return self.predict_cost_rule_based(features, severity_score)
        
        # Add severity score to features
        features_extended = np.append(features, severity_score)
        features_reshaped = features_extended.reshape(1, -1)
        
        cost = self.cost_model.predict(features_reshaped)[0]
        return float(max(cost, 0.0))
    
    def predict_cost(
        self,
        features: np.ndarray,
        severity_score: float
    ) -> Dict[str, float]:
        """
        Predict repair cost.
        
        Args:
            features: Feature array
            severity_score: Severity score
            
        Returns:
            Dictionary with estimated_cost and confidence_interval
        """
        if self.cost_model is not None:
            cost = self.predict_cost_ml_based(features, severity_score)
        else:
            cost = self.predict_cost_rule_based(features, severity_score)
        
        # Estimate confidence interval (±20%)
        lower_bound = cost * 0.8
        upper_bound = cost * 1.2
        
        return {
            "estimated_cost_usd": float(cost),
            "lower_bound_usd": float(lower_bound),
            "upper_bound_usd": float(upper_bound)
        }
    
    def predict_urgency(
        self,
        severity_score: float,
        severity_class: str,
        features: np.ndarray
    ) -> str:
        """
        Predict repair urgency.
        
        Args:
            severity_score: Severity score
            severity_class: Severity classification
            features: Feature array
            
        Returns:
            Urgency level ('immediate', 'urgent', 'scheduled', 'monitor')
        """
        # Extract critical features
        max_depth = features[11] if len(features) > 11 else 0.0
        max_deformation = features[17] if len(features) > 17 else 0.0
        
        # Critical conditions requiring immediate action
        if severity_class == "critical":
            return "immediate"
        
        if max_depth > 20 or max_deformation > 15:
            return "immediate"
        
        # Severe damage requires urgent attention
        if severity_class == "severe":
            return "urgent"
        
        if severity_score > 60:
            return "urgent"
        
        # Moderate damage can be scheduled
        if severity_class == "moderate":
            return "scheduled"
        
        if severity_score > 30:
            return "scheduled"
        
        # Minor damage just needs monitoring
        return "monitor"
    
    def predict_all(
        self,
        features: np.ndarray,
        severity_score: float,
        severity_class: str
    ) -> Dict[str, any]:
        """
        Predict cost and urgency together.
        
        Args:
            features: Feature array
            severity_score: Severity score
            severity_class: Severity class
            
        Returns:
            Complete prediction dictionary
        """
        cost_info = self.predict_cost(features, severity_score)
        urgency = self.predict_urgency(severity_score, severity_class, features)
        
        # Urgency descriptions
        urgency_descriptions = {
            "immediate": "Requires immediate attention - potential safety hazard",
            "urgent": "Repair needed within 1-2 weeks",
            "scheduled": "Schedule repair within 1-2 months",
            "monitor": "Monitor condition, repair when convenient"
        }
        
        return {
            "cost_prediction": cost_info,
            "urgency": urgency,
            "urgency_description": urgency_descriptions[urgency],
            "recommended_timeline": self._get_timeline(urgency)
        }
    
    def _get_timeline(self, urgency: str) -> str:
        """Get recommended timeline for urgency level."""
        timelines = {
            "immediate": "24-48 hours",
            "urgent": "1-2 weeks",
            "scheduled": "1-2 months",
            "monitor": "As needed"
        }
        return timelines.get(urgency, "Unknown")


def create_cost_predictor(
    cost_model_path: Path = None,
    base_cost: float = 100.0
) -> CostPredictor:
    """
    Factory function to create cost predictor.
    
    Args:
        cost_model_path: Path to trained model
        base_cost: Base repair cost
        
    Returns:
        CostPredictor instance
    """
    return CostPredictor(cost_model_path=cost_model_path, base_cost=base_cost)
