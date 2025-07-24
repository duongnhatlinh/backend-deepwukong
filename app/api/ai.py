"""
AI Model API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from app.schemas.common import SuccessResponse
from app.services.deepwukong_service import DeepWuKongService

router = APIRouter()

def get_deepwukong_service() -> DeepWuKongService:
    from app.main import app
    return app.state.deepwukong

@router.get("/status", response_model=SuccessResponse)
async def get_ai_status(
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Get AI model status and information"""
    try:
        status = service.get_status()
        return SuccessResponse(
            data=status,
            message="AI model status retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info", response_model=SuccessResponse)
async def get_ai_info(
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Get detailed AI model information"""
    try:
        status = service.get_status()
        model_info = status.get("model_info", {})
        
        detailed_info = {
            "model_name": "DeepWukong",
            "description": "Deep Graph Neural Network for Software Vulnerability Detection",
            "paper": "TOSEM'21: DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network",
            "supported_languages": ["C", "C++"],
            "supported_extensions": [".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"],
            "vulnerability_types": [
                "Buffer Overflow",
                "NULL Pointer Dereference", 
                "Integer Overflow",
                "Unsafe Function Calls"
            ],
            "current_status": model_info
        }
        
        return SuccessResponse(
            data=detailed_info,
            message="AI model information retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))