"""
Analysis API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from typing import Optional, List
import os
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
async def analyze_files(
    files: List[UploadFile] = File(...),
    name: Optional[str] = Form(None),
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Analyze multiple files for vulnerabilities and return detailed results"""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 50:  # Limit number of files
        raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files allowed")
    
    # Get confidence threshold from database settings
    settings_service = SettingsService()
    confidence_threshold = await settings_service.get_setting("confidence_threshold", 0.7)
    
    # Validate all files first
    validated_files = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail=f"No filename provided for one of the files")
        
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
        
        validated_files.append((content, file.filename))
    
    batch_service = BatchAnalysisService()
    
    try:
        # Use batch analysis service to process multiple files
        batch_id = await batch_service.analyze_multiple_files(
            files=validated_files,
            confidence_threshold=confidence_threshold,
            deepwukong_service=service,
            name=name
        )
        
        # Get the detailed results immediately
        batch_result = await batch_service.get_batch_analysis(batch_id)
        
        if not batch_result:
            raise HTTPException(status_code=500, detail="Failed to retrieve analysis results")
        
        return SuccessResponse(
            data=batch_result,
            message=f"Files analyzed successfully. {batch_result.successful_files} successful, {batch_result.failed_files} failed"
        )
        
    except AnalysisError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

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
