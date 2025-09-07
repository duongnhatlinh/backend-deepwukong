"""
Batch Analysis schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from .analysis import AnalysisResultSchema, AnalysisStatus, VulnerabilitySchema

class BatchAnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"

class FileAnalysisResult(BaseModel):
    file_name: str
    file_path: str
    status: AnalysisStatus
    results: Optional[AnalysisResultSchema] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    
    class Config:
        use_enum_values = True

class BatchAnalysisResult(BaseModel):
    batch_id: str
    name: Optional[str] = None
    status: BatchAnalysisStatus
    total_files: int
    successful_files: int
    failed_files: int
    processing_time_seconds: Optional[float] = None
    file_results: List[FileAnalysisResult]
    
    # Summary statistics
    total_vulnerabilities: int = 0
    total_high_confidence: int = 0
    most_vulnerable_files: List[str] = []
    
    class Config:
        use_enum_values = True

class BatchAnalysisCreateRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=255, description="Tên mô tả cho phân tích batch")
    confidence_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    max_files: Optional[int] = Field(default=50, ge=1, le=100)  # Limit for safety
    
    class Config:
        extra = "forbid"

class BatchAnalysisListItem(BaseModel):
    id: str
    name: Optional[str] = None
    timestamp: datetime
    total_files: int
    successful_files: int
    failed_files: int
    status: BatchAnalysisStatus
    total_vulnerabilities: int
    processing_time_seconds: Optional[float] = None
    
    class Config:
        use_enum_values = True

class BatchAnalysisListResponse(BaseModel):
    batch_analyses: List[BatchAnalysisListItem]
    total: int
    limit: int
    offset: int
class BatchAnalysisDetailedItem(BaseModel):
    """Detailed batch analysis item with full information"""
    id: str
    name: Optional[str] = None
    timestamp: datetime
    total_files: int
    successful_files: int
    failed_files: int
    status: BatchAnalysisStatus
    total_vulnerabilities: int
    total_high_confidence: int
    processing_time_seconds: Optional[float] = None
    confidence_threshold: Optional[float] = None
    source_type: Optional[str] = None
    source_info: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Detailed file results
    file_results: List[FileAnalysisResult] = []
    
    # Summary statistics
    most_vulnerable_files: List[str] = []
    
    class Config:
        use_enum_values = True

class BatchAnalysisDetailedListResponse(BaseModel):
    """Response schema for detailed batch analyses list"""
    batch_analyses: List[BatchAnalysisDetailedItem]
    total: int
    limit: int
    offset: int
