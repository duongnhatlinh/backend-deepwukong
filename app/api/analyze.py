"""
Analysis API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from typing import Optional, List
import os
import tempfile
import aiofiles
from datetime import datetime
import asyncio

from app.schemas.analysis import AnalysisResponse, AnalysisListResponse
from app.schemas.batch_analysis import BatchAnalysisResult, FileAnalysisResult
from app.schemas.common import SuccessResponse
from app.services.deepwukong_service import DeepWuKongService
from app.services.analysis_service import AnalysisService
from app.services.batch_analysis_service import BatchAnalysisService
from app.services.settings_service import SettingsService
from app.core.exceptions import AnalysisError
from app.config import settings

router = APIRouter()

def get_deepwukong_service() -> DeepWuKongService:
    from app.main import app
    return app.state.deepwukong

@router.post("/analyze", response_model=SuccessResponse)
async def analyze_file(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Analyze a single file for vulnerabilities and return detailed results"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Get confidence threshold from database settings
    settings_service = SettingsService()
    confidence_threshold = await settings_service.get_setting("confidence_threshold", 0.7)
    
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported for {file.filename}. Allowed: {settings.allowed_extensions_list}"
        )
    
    # Read and validate file size
    content = await file.read()
    file_size = len(content)
    
    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File {file.filename} too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail=f"File {file.filename} is empty")
    
    # Create temporary file for analysis
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=file_ext) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Prepare analysis options
        analysis_options = {
            "confidence_threshold": confidence_threshold
        }
        
        # Analyze file using DeepWuKong service
        results = await service.analyze_file(temp_file_path, analysis_options)
        
        # Override the file_analyzed field with the original filename
        results["file_analyzed"] = file.filename
        
        # Save analysis results to database
        analysis_service = AnalysisService()
        analysis_id = await analysis_service.save_analysis(
            file_name=file.filename,
            file_path=temp_file_path,
            file_size=file_size,
            results=results,
            confidence_threshold=confidence_threshold,
            name=name,
            model_version=results.get("model_version")
        )
        
        # Add analysis ID to results
        results["analysis_id"] = analysis_id
        
        return SuccessResponse(
            data=results,
            message="File analyzed successfully"
        )
        
    except AnalysisError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass  # Ignore cleanup errors

@router.get("/analyses", response_model=SuccessResponse)
async def list_analyses(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get list of analyses with pagination"""
    try:
        analysis_service = AnalysisService()
        analyses = await analysis_service.get_analyses(limit=limit, offset=offset)
        
        return SuccessResponse(
            data=analyses,
            message="Analyses retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyses/{analysis_id}", response_model=SuccessResponse)
async def get_analysis(analysis_id: str):
    """Get specific analysis by ID"""
    try:
        analysis_service = AnalysisService()
        analysis = await analysis_service.get_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return SuccessResponse(
            data=analysis,
            message="Analysis retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/analyses/{analysis_id}", response_model=SuccessResponse)
async def delete_analysis(analysis_id: str):
    """Delete specific analysis by ID"""
    try:
        analysis_service = AnalysisService()
        deleted = await analysis_service.delete_analysis(analysis_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return SuccessResponse(
            data={"deleted": True},
            message="Analysis deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Thêm vào app/api/ai_analyze.py (cuối file)

@router.get("/analytics", response_model=SuccessResponse)
async def get_ai_analytics():
    """Get AI analytics and statistics"""
    
    # Mock analytics data (trong production sẽ query từ database)
    ai_analytics = {
        "total_ai_scans": 42,
        "total_ai_detections": 127,
        "accuracy_rate": 92.5,
        "average_confidence": 0.83,
        "average_processing_time": 2.34,
        
        # Vulnerability type breakdown from AI
        "vulnerability_types": {
            "buffer_overflow": 45,
            "null_pointer_dereference": 32,
            "integer_overflow": 28,
            "unsafe_function_call": 22
        },
        
        # Confidence distribution
        "confidence_distribution": {
            "high_confidence": 78,  # >= 0.8
            "medium_confidence": 34,  # 0.6-0.8
            "low_confidence": 15   # < 0.6
        },
        
        # Performance metrics
        "performance": {
            "avg_file_size_kb": 15.2,
            "avg_lines_analyzed": 456,
            "fastest_analysis_time": 0.8,
            "slowest_analysis_time": 5.2
        },
        
        # Model information
        "model_info": {
            "version": "DeepWukong-v1.0",
            "architecture": "GCN + RNN",
            "last_updated": "2024-01-15",
            "training_accuracy": 94.2
        }
    }
    
    return SuccessResponse(
        data=ai_analytics,
        message="AI analytics retrieved successfully"
    )
