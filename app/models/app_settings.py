"""
Application Settings database model
"""

from sqlalchemy import Column, String, Float, Text, Boolean
from app.models.base import BaseModel
from app.database import Base

class AppSettings(BaseModel, Base):
    __tablename__ = "app_settings"
    
    # Setting key (unique identifier)
    key = Column(String, nullable=False, unique=True)
    
    # Setting value (stored as text, can be parsed based on type)
    value = Column(Text, nullable=False)
    
    # Setting type for parsing (string, float, int, boolean, json)
    value_type = Column(String, nullable=False, default="string")
    
    # Description of the setting
    description = Column(Text)
    
    # Whether this setting can be modified by users
    user_modifiable = Column(Boolean, default=True)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "value_type": self.value_type,
            "description": self.description,
            "user_modifiable": self.user_modifiable,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def get_typed_value(self):
        """Get the value converted to its proper type"""
        if self.value_type == "float":
            return float(self.value)
        elif self.value_type == "int":
            return int(self.value)
        elif self.value_type == "boolean":
            return self.value.lower() in ("true", "1", "yes", "on")
        elif self.value_type == "json":
            import json
            return json.loads(self.value)
        else:  # string
            return self.value
