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
async def get_ai_model_status(
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Get DeepWukong AI model status"""
    status = service.get_status()
    
    ai_status = {
        "model_loaded": status["model_loaded"],
        "model_ready": status["status"] == "ready",
        "model_version": status["model_info"].get("version", "unknown"),
        "model_architecture": status["model_info"].get("architecture", "unknown"),
        "load_time": status["model_info"].get("load_time", 0),
        "status_message": "DeepWukong AI Ready" if status["model_loaded"] else "DeepWukong AI Not Available",
        "is_mock_mode": status["model_info"].get("status") == "mock"
    }
    
    return SuccessResponse(
        data=ai_status,
        message="AI model status retrieved"
    )


@router.get("/info", response_model=SuccessResponse)
async def get_ai_model_info(
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Get detailed AI model information"""
    status = service.get_status()
    
    ai_info = {
        "model_name": "DeepWukong",
        "description": "Deep Graph Neural Network for Software Vulnerability Detection",
        "paper": "TOSEM'21: DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network",
        "architecture": "GCN + RNN Classifier",
        "supported_languages": ["C", "C++"],
        "supported_extensions": [".c", ".cpp", ".h", ".hpp"],
        "vulnerability_types": [
            "Buffer Overflow",
            "NULL Pointer Dereference",
            "Integer Overflow", 
            "Unsafe Function Calls"
        ],
        "model_details": status["model_info"],
        "confidence_range": "0.1 - 1.0",
        "recommended_threshold": 0.7
    }
    
    return SuccessResponse(
        data=ai_info,
        message="AI model information retrieved"
    )