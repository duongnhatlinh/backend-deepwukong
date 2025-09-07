"""
Batch Analysis API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from typing import List, Optional
import os
import zipfile
import tempfile
import shutil
from datetime import datetime
import asyncio

from app.schemas.batch_analysis import (
    BatchAnalysisResult, BatchAnalysisCreateRequest, BatchAnalysisListResponse, BatchAnalysisDetailedListResponse
)
from app.schemas.common import SuccessResponse
from app.services.deepwukong_service import DeepWuKongService
from app.services.batch_analysis_service import BatchAnalysisService
from app.services.settings_service import SettingsService
from app.core.exceptions import AnalysisError
from app.config import settings

router = APIRouter()

def get_deepwukong_service() -> DeepWuKongService:
    from app.main import app
    return app.state.deepwukong

@router.post("/analyze-multiple", response_model=SuccessResponse)
async def analyze_multiple_files(
    files: List[UploadFile] = File(...),
    name: Optional[str] = Form(None),
    
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Analyze multiple files for vulnerabilities"""
    
    # Validate files count

    # Get confidence threshold from database settings
    settings_service = SettingsService()
    confidence_threshold = await settings_service.get_setting("confidence_threshold", 0.7)
    if len(files) > settings.MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed: {settings.MAX_BATCH_FILES}"
        )
    
    
    # Validate and prepare files
    prepared_files = []
    for file in files:
        if not file.filename:
            continue
            
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.allowed_extensions_list:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} has unsupported extension. Allowed: {settings.allowed_extensions_list}"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is empty"
            )
        
        prepared_files.append((content, file.filename))
    
    if not prepared_files:
        raise HTTPException(status_code=400, detail="No valid files provided")
    
    batch_service = BatchAnalysisService()
    
    try:
        # Start batch analysis
        batch_id = await batch_service.analyze_multiple_files(
            prepared_files, confidence_threshold, service, name
        )
        
        return SuccessResponse(
            data={
                "batch_id": batch_id,
                "name": name,
                "status": "processing",
                "total_files": len(prepared_files),
                "message": f"Batch analysis started for {len(prepared_files)} files"
            },
            message="Batch analysis started successfully"
        )
        
    except AnalysisError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.post("/analyze-zip", response_model=SuccessResponse)
async def analyze_zip_file(
    zip_file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    
    max_files: Optional[int] = Form(50),
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Analyze files from a ZIP archive"""
    
    # Validate ZIP file
    if not zip_file.filename or not zip_file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    # Check ZIP file size
    zip_content = await zip_file.read()
    if len(zip_content) > 100 * 1024 * 1024:  # 100MB limit for ZIP
        raise HTTPException(status_code=400, detail="ZIP file too large (max 100MB)")
    
    # Validate parameters

    # Get confidence threshold from database settings
    settings_service = SettingsService()
    confidence_threshold = await settings_service.get_setting("confidence_threshold", 0.7)
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence threshold must be between 0.0 and 1.0"
        )
    
    if not 1 <= max_files <= 100:
        raise HTTPException(
            status_code=400,
            detail="max_files must be between 1 and 100"
        )
    
    # Create temporary directory for extraction
    temp_dir = tempfile.mkdtemp(prefix="deepwukong_zip_")
    
    try:
        # Save and extract ZIP
        zip_path = os.path.join(temp_dir, "upload.zip")
        with open(zip_path, 'wb') as f:
            f.write(zip_content)
        
        extract_dir = os.path.join(temp_dir, "extracted")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Security check: prevent path traversal
                for member in zip_ref.namelist():
                    if os.path.isabs(member) or ".." in member:
                        raise HTTPException(
                            status_code=400,
                            detail="ZIP contains unsafe file paths"
                        )
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")
        
        # Analyze extracted directory
        batch_service = BatchAnalysisService()
        
        batch_id = await batch_service.analyze_directory(
            extract_dir, confidence_threshold, max_files, service, name
        )
        
        return SuccessResponse(
            data={
                "batch_id": batch_id,
                "name": name,
                "status": "processing",
                "source": "zip_file",
                "zip_filename": zip_file.filename,
                "message": f"ZIP file analysis started with max {max_files} files"
            },
            message="ZIP file analysis started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ZIP analysis failed: {str(e)}")
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp directory {temp_dir}: {e}")

@router.post("/analyze-directory", response_model=SuccessResponse)
async def analyze_directory_path(
    directory_path: str = Form(...),
    name: Optional[str] = Form(None),
    
    max_files: Optional[int] = Form(50),
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Analyze files from a directory path (for server-side directories)"""
    
    # Security: Only allow relative paths or paths within allowed directories
    # This is a security consideration - you might want to restrict this endpoint
    if os.path.isabs(directory_path):
        raise HTTPException(
            status_code=400,
            detail="Absolute paths not allowed for security reasons"
        )
    
    # Validate directory exists
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail="Directory not found")
    
    # Validate parameters

    # Get confidence threshold from database settings
    settings_service = SettingsService()
    confidence_threshold = await settings_service.get_setting("confidence_threshold", 0.7)
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence threshold must be between 0.0 and 1.0"
        )
    
    if not 1 <= max_files <= 100:
        raise HTTPException(
            status_code=400,
            detail="max_files must be between 1 and 100"
        )
    
    batch_service = BatchAnalysisService()
    
    try:
        batch_id = await batch_service.analyze_directory(
            directory_path, confidence_threshold, max_files, service, name
        )
        
        return SuccessResponse(
            data={
                "batch_id": batch_id,
                "name": name,
                "status": "processing",
                "directory_path": directory_path,
                "max_files": max_files,
                "message": f"Directory analysis started for up to {max_files} files"
            },
            message="Directory analysis started successfully"
        )
        
    except AnalysisError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Directory analysis failed: {str(e)}")

@router.get("/batch-analyses", response_model=SuccessResponse)
async def list_batch_analyses(
    limit: int = Query(10, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get list of batch analyses with pagination"""
    try:
        batch_service = BatchAnalysisService()
        batch_analyses = await batch_service.get_batch_analyses(limit=limit, offset=offset)
        
        return SuccessResponse(
            data=batch_analyses,
            message="Batch analyses retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch-analyses/{batch_id}", response_model=SuccessResponse)
async def get_batch_analysis(batch_id: str):
    """Get specific batch analysis results by ID"""
    try:
        batch_service = BatchAnalysisService()
        batch_analysis = await batch_service.get_batch_analysis(batch_id)
        
        if not batch_analysis:
            raise HTTPException(status_code=404, detail="Batch analysis not found")
        
        return SuccessResponse(
            data=batch_analysis,
            message="Batch analysis retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/batch-analyses/{batch_id}", response_model=SuccessResponse)
async def delete_batch_analysis(batch_id: str):
    """Delete specific batch analysis by ID"""
    try:
        batch_service = BatchAnalysisService()
        deleted = await batch_service.delete_batch_analysis(batch_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Batch analysis not found")
        
        return SuccessResponse(
            data={"deleted": True},
            message="Batch analysis deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch-analyses/{batch_id}/status", response_model=SuccessResponse)
async def get_batch_analysis_status(batch_id: str):
    """Get batch analysis status (lightweight endpoint for polling)"""
    try:
        batch_service = BatchAnalysisService()
        batch_analysis = await batch_service.get_batch_analysis(batch_id)
        
        if not batch_analysis:
            raise HTTPException(status_code=404, detail="Batch analysis not found")
        
        # Return only status information
        status_data = {
            "batch_id": batch_id,
            "status": batch_analysis.status,
            "total_files": batch_analysis.total_files,
            "successful_files": batch_analysis.successful_files,
            "failed_files": batch_analysis.failed_files,
            "total_vulnerabilities": batch_analysis.total_vulnerabilities,
            "processing_time_seconds": batch_analysis.processing_time_seconds
        }
        
        return SuccessResponse(
            data=status_data,
            message="Batch analysis status retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/batch-analyses-detailed", response_model=SuccessResponse)
async def get_detailed_batch_analyses(
    limit: int = Query(10, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get detailed list of batch analyses with full information including file results"""
    try:
        batch_service = BatchAnalysisService()
        detailed_batch_analyses = await batch_service.get_detailed_batch_analyses(limit=limit, offset=offset)
        
        return SuccessResponse(
            data=detailed_batch_analyses,
            message="Detailed batch analyses retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
