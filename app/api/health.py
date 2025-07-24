"""
Health check API endpoints
"""

from fastapi import APIRouter, Depends
from app.schemas.common import SuccessResponse, HealthCheck
from app.config import settings
from app.services.deepwukong_service import DeepWuKongService

router = APIRouter()

def get_deepwukong_service() -> DeepWuKongService:
    from app.main import app
    return app.state.deepwukong

@router.get("/health", response_model=SuccessResponse)
async def health_check(
    service: DeepWuKongService = Depends(get_deepwukong_service)
):
    """Health check endpoint"""
    
    # Check AI model status
    ai_status = service.get_status()
    
    # Check components
    components = {
        "database": "healthy",  # Could add actual DB health check
        "ai_model": "healthy" if ai_status["model_loaded"] else "unhealthy",
        "file_storage": "healthy"  # Could add storage checks
    }
    
    # Overall status
    overall_status = "healthy" if all(
        status == "healthy" for status in components.values()
    ) else "unhealthy"
    
    health_data = HealthCheck(
        status=overall_status,
        version=settings.VERSION,
        components=components
    )
    
    return SuccessResponse(
        data=health_data,
        message="Health check completed"
    )