"""
Analysis Service
Manages analysis database operations
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import SessionLocal
from app.models.analysis import Analysis
from app.schemas.analysis import AnalysisListItem, AnalysisListResponse

class AnalysisService:
    def __init__(self):
        self.db = SessionLocal()
    
    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()
    
    async def save_analysis(
        self,
        file_name: str,
        file_path: str,
        file_size: int,
        results: Dict[str, Any],
        confidence_threshold: float,
        model_version: str = None
    ) -> str:
        """Save analysis results to database"""
        try:
            # Extract summary info
            summary = results.get("summary", {})
            vulnerabilities = results.get("vulnerabilities", [])
            
            # Create analysis record
            analysis = Analysis(
                file_name=file_name,
                file_path=file_path,
                file_size=file_size,
                status="completed",
                vulnerabilities_count=summary.get("total_vulnerabilities", 0),
                high_confidence_count=summary.get("high_confidence", 0),
                processing_time_seconds=self._parse_processing_time(results.get("processing_time", "0s")),
                model_version=model_version or results.get("model_version", "unknown"),
                confidence_threshold=confidence_threshold,
                results_json=json.dumps(results),
                started_at=datetime.now(),
                completed_at=datetime.now()
            )
            
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            
            return analysis.id
            
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to save analysis: {str(e)}")
    
    async def save_failed_analysis(
        self,
        file_name: str,
        file_path: str,
        file_size: int,
        error_message: str,
        confidence_threshold: float
    ) -> str:
        """Save failed analysis to database"""
        try:
            analysis = Analysis(
                file_name=file_name,
                file_path=file_path,
                file_size=file_size,
                status="failed",
                error_message=error_message,
                confidence_threshold=confidence_threshold,
                started_at=datetime.now(),
                completed_at=datetime.now()
            )
            
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            
            return analysis.id
            
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to save failed analysis: {str(e)}")
    
    async def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Get analysis by ID"""
        try:
            analysis = self.db.query(Analysis).filter(Analysis.id == analysis_id).first()
            
            if not analysis:
                return None
            
            result = analysis.to_dict()
            
            # Parse results JSON if available
            if analysis.results_json:
                try:
                    result["results"] = json.loads(analysis.results_json)
                except json.JSONDecodeError:
                    result["results"] = None
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to get analysis: {str(e)}")
    
    async def get_analyses(self, limit: int = 10, offset: int = 0) -> AnalysisListResponse:
        """Get list of analyses with pagination"""
        try:
            # Get total count
            total = self.db.query(Analysis).count()
            
            # Get analyses with pagination
            analyses = (
                self.db.query(Analysis)
                .order_by(desc(Analysis.created_at))
                .offset(offset)
                .limit(limit)
                .all()
            )
            
            # Convert to response format
            analysis_items = []
            for analysis in analyses:
                item = AnalysisListItem(
                    id=analysis.id,
                    file_name=analysis.file_name,
                    timestamp=analysis.created_at,
                    vulnerabilities_found=analysis.vulnerabilities_count or 0,
                    status=analysis.status,
                    high_confidence_count=analysis.high_confidence_count or 0,
                    processing_time_seconds=analysis.processing_time_seconds
                )
                analysis_items.append(item)
            
            return AnalysisListResponse(
                analyses=analysis_items,
                total=total,
                limit=limit,
                offset=offset
            )
            
        except Exception as e:
            raise Exception(f"Failed to get analyses: {str(e)}")
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis by ID"""
        try:
            analysis = self.db.query(Analysis).filter(Analysis.id == analysis_id).first()
            
            if not analysis:
                return False
            
            self.db.delete(analysis)
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to delete analysis: {str(e)}")
    
    def _parse_processing_time(self, time_str: str) -> float:
        """Parse processing time string to seconds"""
        try:
            if time_str.endswith('s'):
                return float(time_str[:-1])
            return float(time_str)
        except:
            return 0.0