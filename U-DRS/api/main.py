"""
FastAPI Main Application
REST API for U-DRS damage reconstruction system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Optional
import uuid
from datetime import datetime
import shutil

from api.schemas import (
    DamageAnalysisResponse,
    HealthResponse,
    ErrorResponse
)
from pipeline.inference import create_pipeline
from pipeline.config import config
from utils.logger import setup_logger, get_logger

# Setup logger
logger = setup_logger(config.LOG_FILE, config.LOG_LEVEL)

# Create FastAPI app
app = FastAPI(
    title="U-DRS API",
    description="Universal Damage Reconstruction System - Image-to-3D Damage Analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

# Storage for results
RESULTS_DIR = config.OUTPUTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    logger.info("Starting U-DRS API server...")
    try:
        pipeline = create_pipeline(device=config.DEVICE)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down U-DRS API server...")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "U-DRS API - Universal Damage Reconstruction System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        version="1.0.0",
        device=config.DEVICE,
        models_loaded=pipeline is not None
    )


@app.post("/api/analyze", response_model=DamageAnalysisResponse, tags=["Analysis"])
async def analyze_damage(
    file: UploadFile = File(..., description="Image file to analyze"),
    pixel_to_mm_ratio: float = Form(1.0, description="Calibration ratio (pixels to mm)"),
    depth_scale: float = Form(1.0, description="Depth scale factor"),
    generate_3d: bool = Form(True, description="Generate 3D reconstruction")
):
    """
    Analyze damage from uploaded image.
    
    Args:
        file: Image file (JPG, PNG, BMP)
        pixel_to_mm_ratio: Calibration for 2D measurements
        depth_scale: Calibration for 3D measurements
        generate_3d: Whether to generate 3D reconstruction
        
    Returns:
        Complete damage analysis results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {config.ALLOWED_EXTENSIONS}"
        )
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now()
    
    # Save uploaded file
    upload_dir = RESULTS_DIR / analysis_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / f"input{file_ext}"
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing analysis {analysis_id}: {file.filename}")
        
        # Update pipeline calibration
        pipeline.pixel_to_mm_ratio = pixel_to_mm_ratio
        pipeline.depth_scale = depth_scale
        pipeline.analyzer.geometric.pixel_to_mm_ratio = pixel_to_mm_ratio
        pipeline.analyzer.volumetric.depth_scale = depth_scale
        
        # Process image
        results = pipeline.process(
            image_path=input_path,
            save_outputs=True,
            output_dir=upload_dir,
            generate_3d=generate_3d
        )
        
        # Add metadata
        results["analysis_id"] = analysis_id
        results["timestamp"] = timestamp
        
        logger.info(f"Analysis {analysis_id} completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing analysis {analysis_id}: {e}")
        # Clean up on error
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/result/{analysis_id}", tags=["Results"])
async def get_result(analysis_id: str):
    """
    Retrieve analysis results by ID.
    
    Args:
        analysis_id: Unique analysis identifier
        
    Returns:
        JSON report
    """
    result_dir = RESULTS_DIR / analysis_id
    report_path = result_dir / "report.json"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    import json
    with open(report_path, "r") as f:
        results = json.load(f)
    
    return results


@app.get("/api/download/{analysis_id}/{file_type}", tags=["Results"])
async def download_file(
    analysis_id: str,
    file_type: str
):
    """
    Download specific result file.
    
    Args:
        analysis_id: Unique analysis identifier
        file_type: File type (mask, depth_map, overlay, point_cloud, mesh, report)
        
    Returns:
        File download
    """
    result_dir = RESULTS_DIR / analysis_id
    
    file_mapping = {
        "mask": "mask.png",
        "depth_map": "depth_map.png",
        "overlay": "overlay.png",
        "edges": "edges.png",
        "point_cloud": "point_cloud.ply",
        "mesh": "mesh.ply",
        "report": "report.json"
    }
    
    if file_type not in file_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid file_type. Options: {list(file_mapping.keys())}")
    
    file_path = result_dir / file_mapping[file_type]
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {file_type} not found for analysis {analysis_id}")
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream"
    )


@app.get("/api/list", tags=["Results"])
async def list_analyses():
    """
    List all available analyses.
    
    Returns:
        List of analysis IDs and metadata
    """
    analyses = []
    
    for analysis_dir in RESULTS_DIR.iterdir():
        if analysis_dir.is_dir():
            report_path = analysis_dir / "report.json"
            if report_path.exists():
                import json
                with open(report_path, "r") as f:
                    report = json.load(f)
                
                analyses.append({
                    "analysis_id": analysis_dir.name,
                    "timestamp": report.get("timestamp"),
                    "status": report.get("status"),
                    "severity": report.get("severity", {}).get("class"),
                    "score": report.get("severity", {}).get("score")
                })
    
    # Sort by timestamp (newest first)
    analyses.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return {"analyses": analyses, "count": len(analyses)}


@app.delete("/api/result/{analysis_id}", tags=["Results"])
async def delete_analysis(analysis_id: str):
    """
    Delete analysis results.
    
    Args:
        analysis_id: Analysis ID to delete
        
    Returns:
        Success message
    """
    result_dir = RESULTS_DIR / analysis_id
    
    if not result_dir.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    shutil.rmtree(result_dir)
    logger.info(f"Deleted analysis {analysis_id}")
    
    return {"message": f"Analysis {analysis_id} deleted successfully"}


# Mount static files if UI exists
# app.mount("/static", StaticFiles(directory="static"), name="static")


def start_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False
):
    """
    Start FastAPI server.
    
    Args:
        host: Host address
        port: Port number
        reload: Enable auto-reload
    """
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_server(reload=True)
