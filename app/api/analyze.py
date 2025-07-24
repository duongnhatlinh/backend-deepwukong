"""
Analysis API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from typing import Optional
import os
import aiofiles
from datetime import datetime
import asyncio

from app.schemas.analysis import AnalysisResponse, AnalysisListResponse
from app.schemas.common import SuccessResponse
from app.services.deepwukong_service import DeepWuKongService
from app.services.analysis_service import AnalysisService
from app.core.exceptions import AnalysisError
from app.config import settings

router = APIRouter()

def get_deepwukong_service() -> DeepWuKongService:
    from app.main import app
    return app.state.deepwukong

@router.post("/analyze", response_model=SuccessResponse)
async def analyze_file(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(0.7),
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Analyze a single file for vulnerabilities"""
    
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed: {settings.allowed_extensions_list}"
        )
    
    # Read and validate file size
    content = await file.read()
    file_size = len(content)
    
    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    # Generate unique filename to prevent conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_filename = f"{timestamp}_{file.filename}"
    temp_file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
    
    analysis_service = AnalysisService()
    
    try:
        # Save uploaded file temporarily
        async with aiofiles.open(temp_file_path, 'wb') as f:
            await f.write(content)
        
        # Validate confidence threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise HTTPException(
                status_code=400, 
                detail="Confidence threshold must be between 0.0 and 1.0"
            )
        
        # Analyze file with timeout
        analysis_options = {
            "confidence_threshold": confidence_threshold
        }
        
        try:
            results = await asyncio.wait_for(
                service.analyze_file(temp_file_path, analysis_options),
                timeout=settings.AI_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            # Save failed analysis
            analysis_id = await analysis_service.save_failed_analysis(
                file_name=file.filename,
                file_path=temp_file_path,
                file_size=file_size,
                error_message="Analysis timeout",
                confidence_threshold=confidence_threshold
            )
            raise HTTPException(
                status_code=408, 
                detail=f"Analysis timeout after {settings.AI_TIMEOUT_SECONDS} seconds"
            )
        
        # Save successful analysis to database
        analysis_id = await analysis_service.save_analysis(
            file_name=file.filename,
            file_path=temp_file_path,
            file_size=file_size,
            results=results,
            confidence_threshold=confidence_threshold,
            model_version=results.get("model_version")
        )
        
        return SuccessResponse(
            data={
                "analysis_id": analysis_id,
                "status": "completed",
                "results": results
            },
            message="File analyzed successfully"
        )
        
    except HTTPException:
        raise
    except AnalysisError as e:
        # Save failed analysis
        analysis_id = await analysis_service.save_failed_analysis(
            file_name=file.filename,
            file_path=temp_file_path,
            file_size=file_size,
            error_message=str(e),
            confidence_threshold=confidence_threshold
        )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Save failed analysis
        analysis_id = await analysis_service.save_failed_analysis(
            file_name=file.filename,
            file_path=temp_file_path,
            file_size=file_size,
            error_message=f"Unexpected error: {str(e)}",
            confidence_threshold=confidence_threshold
        )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp file {temp_file_path}: {e}")

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