"""
Settings schemas
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class SettingValue(BaseModel):
    """Setting value schema"""
    value: Any
    value_type: str
    description: Optional[str] = None
    user_modifiable: bool = True

class SettingsResponse(BaseModel):
    """Response schema for settings"""
    settings: Dict[str, SettingValue]
    
    class Config:
        from_attributes = True

class UpdateSettingRequest(BaseModel):
    """Request schema for updating a setting"""
    value: Any
    value_type: Optional[str] = None
    
    class Config:
        extra = "forbid"

class UpdateSettingsRequest(BaseModel):
    """Request schema for updating multiple settings"""
    settings: Dict[str, UpdateSettingRequest]
    
    class Config:
        extra = "forbid"

class DeleteSettingsRequest(BaseModel):
    """Request schema for deleting multiple settings"""
    keys: List[str]
    
    class Config:
        extra = "forbid"
