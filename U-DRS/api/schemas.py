"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


class DamageDetectionRequest(BaseModel):
    """Request model for damage detection."""
    pixel_to_mm_ratio: Optional[float] = Field(1.0, description="Calibration ratio from pixels to mm")
    depth_scale: Optional[float] = Field(1.0, description="Depth scale factor")
    generate_3d: Optional[bool] = Field(True, description="Generate 3D reconstruction")
    

class BoundingBox(BaseModel):
    """Bounding box model."""
    width: float
    height: float
    area: float
    aspect_ratio: float
    x_min: Optional[int] = None
    y_min: Optional[int] = None
    x_max: Optional[int] = None
    y_max: Optional[int] = None


class CrackWidth(BaseModel):
    """Crack width statistics."""
    mean: float
    max: float
    min: float
    std: float


class GeometricMeasurements(BaseModel):
    """2D geometric measurements."""
    area_mm2: float
    perimeter_mm: float
    crack_length_mm: float
    crack_width: CrackWidth
    bounding_box: BoundingBox
    compactness: float
    circularity: float
    num_regions: int


class DepthStats(BaseModel):
    """Depth statistics."""
    max_mm: float
    mean_mm: float
    std_mm: float
    range_mm: float


class Deformation(BaseModel):
    """Deformation metrics."""
    deformation_volume_mm3: float
    mean_deformation_mm: float
    max_deformation_mm: float


class VolumetricMeasurements(BaseModel):
    """3D volumetric measurements."""
    depth_stats: Optional[DepthStats] = None
    volume_from_depth_mm3: Optional[float] = None
    deformation: Optional[Deformation] = None
    volume_from_mesh_mm3: Optional[float] = None
    surface_area_mm2: Optional[float] = None


class MeasurementSummary(BaseModel):
    """Summary of key measurements."""
    damage_area_mm2: float
    crack_length_mm: float
    crack_width_mean_mm: float
    crack_width_max_mm: float
    bbox_width_mm: float
    bbox_height_mm: float
    max_depth_mm: Optional[float] = None
    mean_depth_mm: Optional[float] = None
    volume_mm3: Optional[float] = None
    max_deformation_mm: Optional[float] = None


class MeasurementResult(BaseModel):
    """Complete measurement results."""
    geometric_2d: GeometricMeasurements = Field(..., alias="2d_measurements")
    volumetric_3d: Optional[VolumetricMeasurements] = Field(None, alias="3d_measurements")
    summary: MeasurementSummary
    
    class Config:
        populate_by_name = True


class SeverityResult(BaseModel):
    """Severity prediction result."""
    score: float = Field(..., description="Severity score 0-100")
    class_: str = Field(..., alias="class", description="Severity class")
    confidence: float = Field(..., description="Confidence in classification")
    threshold_low: float
    threshold_high: float
    
    class Config:
        populate_by_name = True


class CostPrediction(BaseModel):
    """Cost prediction result."""
    estimated_cost_usd: float
    lower_bound_usd: float
    upper_bound_usd: float


class CostUrgencyResult(BaseModel):
    """Cost and urgency prediction."""
    cost_prediction: CostPrediction
    urgency: str = Field(..., description="Urgency level")
    urgency_description: str
    recommended_timeline: str


class TimingInfo(BaseModel):
    """Processing timing information."""
    detection: float
    segmentation: float
    postprocessing: float
    depth_estimation: float
    reconstruction_3d: Optional[float] = Field(None, alias="3d_reconstruction")
    measurements: float
    severity_scoring: float
    cost_prediction: float
    total: float
    
    class Config:
        populate_by_name = True


class DetectionInfo(BaseModel):
    """Detection result info."""
    is_damaged: bool
    confidence: float
    probabilities: List[float]


class DamageAnalysisResponse(BaseModel):
    """Complete damage analysis response."""
    status: str = Field(..., description="Processing status")
    input_image: str
    detection: Optional[DetectionInfo] = None
    measurements: Optional[MeasurementResult] = None
    severity: Optional[SeverityResult] = None
    cost_urgency: Optional[CostUrgencyResult] = None
    timing: TimingInfo
    output_dir: str
    analysis_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    device: str
    models_loaded: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
