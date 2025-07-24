"""
Analysis schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class VulnerabilityType(str, Enum):
    BUFFER_OVERFLOW = "buffer_overflow"
    NULL_POINTER_DEREFERENCE = "null_pointer_dereference"
    INTEGER_OVERFLOW = "integer_overflow"
    UNSAFE_FUNCTION_CALL = "unsafe_function_call"
    UNKNOWN_VULNERABILITY = "unknown_vulnerability"

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VulnerabilitySchema(BaseModel):
    line_number: int
    type: VulnerabilityType
    confidence: float = Field(ge=0.0, le=1.0)
    severity: SeverityLevel
    description: str
    code_snippet: Optional[str] = None
    api_type: str  # call/array/ptr/arith
    recommendation: Optional[str] = None
    
    class Config:
        use_enum_values = True

class SummarySchema(BaseModel):
    total_vulnerabilities: int
    high_confidence: int
    medium_confidence: int
    low_confidence: int
    processing_time: str
    severity_counts: Dict[str, int]

class AnalysisResultSchema(BaseModel):
    file_analyzed: str
    vulnerabilities: List[VulnerabilitySchema]
    summary: SummarySchema
    processing_time: str
    model_version: str

class AnalysisCreateRequest(BaseModel):
    confidence_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    
    class Config:
        extra = "forbid"

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    results: Optional[AnalysisResultSchema] = None
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True

class AnalysisListItem(BaseModel):
    id: str
    file_name: str
    timestamp: datetime
    vulnerabilities_found: int
    status: AnalysisStatus
    high_confidence_count: int
    processing_time_seconds: Optional[float] = None
    
    class Config:
        use_enum_values = True

class AnalysisListResponse(BaseModel):
    analyses: List[AnalysisListItem]
    total: int
    limit: int
    offset: int