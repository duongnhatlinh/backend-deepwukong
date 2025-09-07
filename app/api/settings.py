"""
Settings API endpoints
"""

from fastapi import APIRouter, HTTPException
from app.services.settings_service import SettingsService
from app.schemas.settings import SettingsResponse, UpdateSettingRequest, UpdateSettingsRequest, DeleteSettingsRequest
from app.schemas.common import SuccessResponse

router = APIRouter()

@router.get("/settings", response_model=SuccessResponse)
async def get_all_settings():
    """Get all application settings"""
    try:
        settings_service = SettingsService()
        settings = await settings_service.get_all_settings()
        
        return SuccessResponse(
            data={"settings": settings},
            message="Settings retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/settings/user-modifiable", response_model=SuccessResponse)
async def get_user_modifiable_settings():
    """Get only user-modifiable settings"""
    try:
        settings_service = SettingsService()
        settings = await settings_service.get_user_modifiable_settings()
        
        return SuccessResponse(
            data={"settings": settings},
            message="User-modifiable settings retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/settings/{key}", response_model=SuccessResponse)
async def get_setting(key: str):
    """Get a specific setting by key"""
    try:
        settings_service = SettingsService()
        value = await settings_service.get_setting(key)
        
        if value is None:
            raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
        
        return SuccessResponse(
            data={"key": key, "value": value},
            message=f"Setting '{key}' retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/settings/{key}", response_model=SuccessResponse)
async def update_setting(key: str, request: UpdateSettingRequest):
    """Update a specific setting"""
    try:
        settings_service = SettingsService()
        
        # Validate confidence_threshold specifically
        if key == "confidence_threshold":
            if not isinstance(request.value, (int, float)) or not (0.0 <= request.value <= 1.0):
                raise HTTPException(
                    status_code=400,
                    detail="Confidence threshold must be a number between 0.0 and 1.0"
                )
        
        success = await settings_service.set_setting(
            key=key,
            value=request.value,
            value_type=request.value_type or "string"
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update setting")
        
        return SuccessResponse(
            data={"key": key, "value": request.value},
            message=f"Setting '{key}' updated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/settings", response_model=SuccessResponse)
async def update_multiple_settings(request: UpdateSettingsRequest):
    """Update multiple settings at once"""
    try:
        settings_service = SettingsService()
        updated_settings = {}
        
        for key, setting_request in request.settings.items():
            # Validate confidence_threshold specifically
            if key == "confidence_threshold":
                if not isinstance(setting_request.value, (int, float)) or not (0.0 <= setting_request.value <= 1.0):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Confidence threshold must be a number between 0.0 and 1.0"
                    )
            
            success = await settings_service.set_setting(
                key=key,
                value=setting_request.value,
                value_type=setting_request.value_type or "string"
            )
            
            if success:
                updated_settings[key] = setting_request.value
            else:
                raise HTTPException(status_code=500, detail=f"Failed to update setting '{key}'")
        
        return SuccessResponse(
            data={"updated_settings": updated_settings},
            message=f"Updated {len(updated_settings)} settings successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/settings/{key}", response_model=SuccessResponse)
async def delete_setting(key: str):
    """Delete a specific setting"""
    try:
        settings_service = SettingsService()
        
        # Prevent deletion of critical settings
        critical_settings = ["confidence_threshold", "max_file_size_mb", "max_batch_files"]
        if key in critical_settings:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete critical setting '{key}'. Critical settings: {critical_settings}"
            )
        
        success = await settings_service.delete_setting(key)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Setting '{key}' not found or could not be deleted")
        
        return SuccessResponse(
            data={"deleted_key": key},
            message=f"Setting '{key}' deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/settings", response_model=SuccessResponse)
async def delete_multiple_settings(request: DeleteSettingsRequest):
    """Delete multiple settings at once"""
    try:
        settings_service = SettingsService()
        
        # Prevent deletion of critical settings
        critical_settings = ["confidence_threshold", "max_file_size_mb", "max_batch_files"]
        critical_in_request = [key for key in request.keys if key in critical_settings]
        
        if critical_in_request:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete critical settings: {critical_in_request}. Critical settings: {critical_settings}"
            )
        
        results = await settings_service.delete_multiple_settings(request.keys)
        
        deleted_keys = [key for key, success in results.items() if success]
        failed_keys = [key for key, success in results.items() if not success]
        
        if failed_keys:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to delete settings: {failed_keys}"
            )
        
        return SuccessResponse(
            data={"deleted_keys": deleted_keys},
            message=f"Deleted {len(deleted_keys)} settings successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
