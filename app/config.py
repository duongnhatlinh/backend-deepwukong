"""
Application Configuration
"""

from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "DeepWukong API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Database
    DATABASE_URL: str = "sqlite:///./storage/deepwukong.db"
    
    # File Storage
    UPLOAD_DIR: str = "./storage/uploads"
    MODELS_DIR: str = "./storage/models"
    RESULTS_DIR: str = "./storage/results"
    LOGS_DIR: str = "./storage/logs"
    
    # File Limits
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = [".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"]
    
    # DeepWukong
    DEEPWUKONG_MODEL_PATH: str = "./storage/models/deepwukong_current.ckpt"
    DEEPWUKONG_CONFIG_PATH: str = "./deepwukong/configs/dwk.yaml"
    JOERN_PATH: str = "./deepwukong/joern/joern-parse"
    
    # AI Configuration
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7
    AI_TIMEOUT_SECONDS: int = 300
    
    # Redis (optional)
    REDIS_URL: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        for directory in [
            self.UPLOAD_DIR,
            self.MODELS_DIR, 
            self.RESULTS_DIR,
            self.LOGS_DIR
        ]:
            os.makedirs(directory, exist_ok=True)

# Create settings instance
settings = Settings()