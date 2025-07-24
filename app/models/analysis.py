"""
Analysis database models
"""

from sqlalchemy import Column, String, Integer, Text, Float, DateTime
from app.models.base import BaseModel
from app.database import Base

class Analysis(BaseModel, Base):
    __tablename__ = "analyses"
    
    # Basic info
    file_name = Column(String, nullable=False)
    file_path = Column(String)
    file_size = Column(Integer)  # bytes
    
    # Status
    status = Column(String, nullable=False, default="pending")  # pending/processing/completed/failed
    
    # Results
    vulnerabilities_count = Column(Integer, default=0)
    high_confidence_count = Column(Integer, default=0)
    processing_time_seconds = Column(Float)
    
    # AI Model info
    model_version = Column(String)
    confidence_threshold = Column(Float, default=0.7)
    
    # Results data (JSON)
    results_json = Column(Text)  # Store full results as JSON string
    
    # Error handling
    error_message = Column(Text)
    
    # Timestamps
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "status": self.status,
            "vulnerabilities_count": self.vulnerabilities_count,
            "high_confidence_count": self.high_confidence_count,
            "processing_time_seconds": self.processing_time_seconds,
            "model_version": self.model_version,
            "confidence_threshold": self.confidence_threshold,
            "results_json": self.results_json,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }