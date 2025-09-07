"""
Settings Service
Manages application settings stored in database
"""

from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.app_settings import AppSettings

class SettingsService:
    def __init__(self):
        self.db = SessionLocal()
    
    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()
    
    async def get_setting(self, key: str, default_value: Any = None) -> Any:
        """Get a setting value by key"""
        try:
            setting = self.db.query(AppSettings).filter(AppSettings.key == key).first()
            
            if not setting:
                return default_value
            
            return setting.get_typed_value()
            
        except Exception as e:
            print(f"Error getting setting {key}: {e}")
            return default_value
    
    async def set_setting(self, key: str, value: Any, value_type: str = "string", 
                         description: str = None, user_modifiable: bool = True) -> bool:
        """Set a setting value"""
        try:
            # Convert value to string for storage
            if value_type == "json":
                import json
                str_value = json.dumps(value)
            else:
                str_value = str(value)
            
            # Check if setting exists
            existing_setting = self.db.query(AppSettings).filter(AppSettings.key == key).first()
            
            if existing_setting:
                # Update existing setting
                existing_setting.value = str_value
                existing_setting.value_type = value_type
                if description:
                    existing_setting.description = description
                existing_setting.user_modifiable = user_modifiable
            else:
                # Create new setting
                new_setting = AppSettings(
                    key=key,
                    value=str_value,
                    value_type=value_type,
                    description=description,
                    user_modifiable=user_modifiable
                )
                self.db.add(new_setting)
            
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            print(f"Error setting {key}: {e}")
            return False
    
    async def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as a dictionary"""
        try:
            settings = self.db.query(AppSettings).all()
            result = {}
            
            for setting in settings:
                result[setting.key] = {
                    "value": setting.get_typed_value(),
                    "value_type": setting.value_type,
                    "description": setting.description,
                    "user_modifiable": setting.user_modifiable
                }
            
            return result
            
        except Exception as e:
            print(f"Error getting all settings: {e}")
            return {}
    
    async def get_user_modifiable_settings(self) -> Dict[str, Any]:
        """Get only user-modifiable settings"""
        try:
            settings = self.db.query(AppSettings).filter(AppSettings.user_modifiable == True).all()
            result = {}
            
            for setting in settings:
                result[setting.key] = {
                    "value": setting.get_typed_value(),
                    "value_type": setting.value_type,
                    "description": setting.description
                }
            
            return result
            
        except Exception as e:
            print(f"Error getting user modifiable settings: {e}")
            return {}
    
    async def initialize_default_settings(self):
        """Initialize default settings if they don't exist"""
        default_settings = [
            {
                "key": "confidence_threshold",
                "value": 0.7,
                "value_type": "float",
                "description": "Default confidence threshold for vulnerability detection (0.0 - 1.0)",
                "user_modifiable": True
            },
            {
                "key": "max_file_size_mb",
                "value": 10,
                "value_type": "int", 
                "description": "Maximum file size in MB for analysis",
                "user_modifiable": True
            },
            {
                "key": "max_batch_files",
                "value": 50,
                "value_type": "int",
                "description": "Maximum number of files in batch analysis",
                "user_modifiable": True
            }
        ]
        
        for setting_data in default_settings:
            existing = self.db.query(AppSettings).filter(AppSettings.key == setting_data["key"]).first()
            if not existing:
                await self.set_setting(**setting_data)

    async def delete_setting(self, key: str) -> bool:
        """Delete a setting by key"""
        try:
            setting = self.db.query(AppSettings).filter(AppSettings.key == key).first()
            
            if not setting:
                return False
            
            # Prevent deletion of critical settings
            critical_settings = ["confidence_threshold", "max_file_size_mb", "max_batch_files"]
            if key in critical_settings:
                print(f"Warning: Attempted to delete critical setting '{key}'")
                return False
            
            self.db.delete(setting)
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            print(f"Error deleting setting {key}: {e}")
            return False
    
    async def delete_multiple_settings(self, keys: list) -> Dict[str, bool]:
        """Delete multiple settings by keys"""
        try:
            results = {}
            critical_settings = ["confidence_threshold", "max_file_size_mb", "max_batch_files"]
            
            for key in keys:
                if key in critical_settings:
                    results[key] = False
                    print(f"Warning: Skipped deletion of critical setting '{key}'")
                    continue
                
                setting = self.db.query(AppSettings).filter(AppSettings.key == key).first()
                if setting:
                    self.db.delete(setting)
                    results[key] = True
                else:
                    results[key] = False
            
            self.db.commit()
            return results
            
        except Exception as e:
            self.db.rollback()
            print(f"Error deleting multiple settings: {e}")
            return {key: False for key in keys}
