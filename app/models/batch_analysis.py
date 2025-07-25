"""
Batch Analysis database models
"""

from sqlalchemy import Column, String, Integer, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.models.base import BaseModel
from app.database import Base

class BatchAnalysis(BaseModel, Base):
    __tablename__ = "batch_analyses"
    
    # Basic info
    total_files = Column(Integer, nullable=False, default=0)
    successful_files = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)
    
    # Status
    status = Column(String, nullable=False, default="pending")  # pending/processing/completed/partial_success/failed
    
    # Results summary
    total_vulnerabilities = Column(Integer, default=0)
    total_high_confidence = Column(Integer, default=0)
    processing_time_seconds = Column(Float)
    
    # Configuration
    confidence_threshold = Column(Float, default=0.7)
    max_files = Column(Integer, default=50)
    
    # Metadata
    source_type = Column(String)  # 'files' or 'directory'
    source_info = Column(Text)  # JSON string with source details
    
    # Error handling
    error_message = Column(Text)
    
    # Timestamps
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationship to individual file analyses
    file_analyses = relationship("Analysis", backref="batch_analysis", foreign_keys="Analysis.batch_analysis_id")
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "status": self.status,
            "total_vulnerabilities": self.total_vulnerabilities,
            "total_high_confidence": self.total_high_confidence,
            "processing_time_seconds": self.processing_time_seconds,
            "confidence_threshold": self.confidence_threshold,
            "max_files": self.max_files,
            "source_type": self.source_type,
            "source_info": self.source_info,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }