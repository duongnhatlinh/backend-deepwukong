"""
Common Pydantic schemas
"""

from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime

class BaseResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    timestamp: datetime = datetime.now()

class SuccessResponse(BaseResponse):
    success: bool = True
    data: Any

class ErrorResponse(BaseResponse):
    success: bool = False
    error: str
    details: Optional[str] = None
    status_code: int = 500

class HealthCheck(BaseModel):
    status: str = "healthy"
    version: str
    timestamp: datetime = datetime.now()
    components: dict = {}

class PaginationParams(BaseModel):
    limit: int = 10
    offset: int = 0
    
    class Config:
        extra = "forbid"