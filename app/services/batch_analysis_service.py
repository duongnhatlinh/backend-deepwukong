"""
Batch Analysis Service
Manages batch analysis operations and coordination
"""

import json
import os
import tempfile
import shutil
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pathlib import Path

from app.database import SessionLocal
from app.models.batch_analysis import BatchAnalysis
from app.models.analysis import Analysis
from app.schemas.batch_analysis import (
    BatchAnalysisResult, FileAnalysisResult, BatchAnalysisListItem, 
    BatchAnalysisListResponse, BatchAnalysisStatus
)
from app.schemas.analysis import AnalysisStatus
from app.services.analysis_service import AnalysisService
from app.services.deepwukong_service import DeepWuKongService
from app.core.exceptions import AnalysisError
from app.config import settings

class BatchAnalysisService:
    def __init__(self):
        self.db = SessionLocal()
        self.analysis_service = AnalysisService()
    
    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()
    
    async def analyze_multiple_files(
        self, 
        files: List[tuple],  # List of (file_content, filename) tuples
        confidence_threshold: float,
        deepwukong_service: DeepWuKongService,
        name: str = None
    ) -> str:
        """
        Analyze multiple files
        Args:
            files: List of (file_content, filename) tuples  
            confidence_threshold: Confidence threshold for analysis
            deepwukong_service: DeepWukong service instance
        Returns:
            batch_analysis_id: ID of the batch analysis
        """
        # Create batch analysis record
        batch_analysis = BatchAnalysis(
            name=name,
            total_files=len(files),
            status="processing",
            confidence_threshold=confidence_threshold,
            source_type="files",
            source_info=json.dumps([filename for _, filename in files]),
            started_at=datetime.now()
        )
        
        self.db.add(batch_analysis)
        self.db.commit()
        self.db.refresh(batch_analysis)
        
        batch_id = batch_analysis.id
        
        try:
            # Process files
            file_results = await self._process_files_batch(
                files, batch_id, confidence_threshold, deepwukong_service
            )
            
            # Update batch analysis with results
            await self._finalize_batch_analysis(batch_id, file_results)
            
            return batch_id
            
        except Exception as e:
            # Mark batch as failed
            await self._mark_batch_failed(batch_id, str(e))
            raise AnalysisError(f"Batch analysis failed: {str(e)}")
    
    async def analyze_directory(
        self,
        directory_path: str,
        confidence_threshold: float,
        max_files: int,
        deepwukong_service: DeepWuKongService,
        name: str = None
    ) -> str:
        """
        Analyze all files in a directory
        Args:
            directory_path: Path to directory containing files
            confidence_threshold: Confidence threshold for analysis
            max_files: Maximum number of files to process
            deepwukong_service: DeepWukong service instance
        Returns:
            batch_analysis_id: ID of the batch analysis
        """
        # Find all valid files in directory
        valid_files = self._find_valid_files(directory_path, max_files)
        
        if not valid_files:
            raise AnalysisError("No valid files found in directory")
        
        # Create batch analysis record
        batch_analysis = BatchAnalysis(
            name=name,
            total_files=len(valid_files),
            status="processing",
            confidence_threshold=confidence_threshold,
            max_files=max_files,
            source_type="directory",
            source_info=json.dumps({
                "directory_path": directory_path,
                "files_found": [os.path.basename(f) for f in valid_files]
            }),
            started_at=datetime.now()
        )
        
        self.db.add(batch_analysis)
        self.db.commit()
        self.db.refresh(batch_analysis)
        
        batch_id = batch_analysis.id
        
        try:
            # Read file contents
            files = []
            for file_path in valid_files:
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    files.append((content, os.path.basename(file_path)))
                except Exception as e:
                    print(f"Warning: Could not read file {file_path}: {e}")
                    continue
            
            # Process files
            file_results = await self._process_files_batch(
                files, batch_id, confidence_threshold, deepwukong_service
            )
            
            # Update batch analysis with results
            await self._finalize_batch_analysis(batch_id, file_results)
            
            return batch_id
            
        except Exception as e:
            # Mark batch as failed
            await self._mark_batch_failed(batch_id, str(e))
            raise AnalysisError(f"Directory analysis failed: {str(e)}")
    
    def _find_valid_files(self, directory_path: str, max_files: int) -> List[str]:
        """Find valid C/C++ files in directory"""
        valid_files = []
        allowed_extensions = settings.allowed_extensions_list
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if len(valid_files) >= max_files:
                    break
                    
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in allowed_extensions:
                    # Check file size
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size <= settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                            valid_files.append(file_path)
                    except OSError:
                        continue
            
            if len(valid_files) >= max_files:
                break
        
        return valid_files
    
    async def _process_files_batch(
        self,
        files: List[tuple],
        batch_id: str,
        confidence_threshold: float,
        deepwukong_service: DeepWuKongService
    ) -> List[FileAnalysisResult]:
        """Process a batch of files"""
        file_results = []
        
        # Process files with controlled concurrency
        semaphore = asyncio.Semaphore(2)  # Limit concurrent processing
        
        async def process_single_file(file_content: bytes, filename: str) -> FileAnalysisResult:
            async with semaphore:
                return await self._process_single_file(
                    file_content, filename, batch_id, confidence_threshold, deepwukong_service
                )
        
        # Create tasks for all files
        tasks = [
            process_single_file(content, filename) 
            for content, filename in files
        ]
        
        # Process all files
        file_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(file_results):
            if isinstance(result, Exception):
                filename = files[i][1]
                error_result = FileAnalysisResult(
                    file_name=filename,
                    file_path=filename,
                    status=AnalysisStatus.FAILED,
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_file(
        self,
        file_content: bytes,
        filename: str,
        batch_id: str,
        confidence_threshold: float,
        deepwukong_service: DeepWuKongService
    ) -> FileAnalysisResult:
        """Process a single file within a batch"""
        file_size = len(file_content)
        
        # Generate unique temp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_filename = f"{timestamp}_{filename}"
        temp_file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
        
        try:
            # Save file temporarily
            with open(temp_file_path, 'wb') as f:
                f.write(file_content)
            
            # Analyze file
            start_time = datetime.now()
            
            analysis_options = {
                "confidence_threshold": confidence_threshold
            }
            
            try:
                results = await asyncio.wait_for(
                    deepwukong_service.analyze_file(temp_file_path, analysis_options),
                    timeout=settings.AI_TIMEOUT_SECONDS
                )
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Save individual analysis to database
                analysis_id = await self.analysis_service.save_analysis(
                    file_name=filename,
                    file_path=temp_file_path,
                    file_size=file_size,
                    results=results,
                    confidence_threshold=confidence_threshold,
                    model_version=results.get("model_version"),
                    batch_analysis_id=batch_id
                )
                
                return FileAnalysisResult(
                    file_name=filename,
                    file_path=filename,
                    status=AnalysisStatus.COMPLETED,
                    results=results,
                    processing_time_seconds=processing_time
                )
                
            except asyncio.TimeoutError:
                # Save failed analysis
                await self.analysis_service.save_failed_analysis(
                    file_name=filename,
                    file_path=temp_file_path,
                    file_size=file_size,
                    error_message="Analysis timeout",
                    confidence_threshold=confidence_threshold,
                    batch_analysis_id=batch_id
                )
                
                return FileAnalysisResult(
                    file_name=filename,
                    file_path=filename,
                    status=AnalysisStatus.FAILED,
                    error_message=f"Analysis timeout after {settings.AI_TIMEOUT_SECONDS} seconds"
                )
                
        except Exception as e:
            # Save failed analysis
            await self.analysis_service.save_failed_analysis(
                file_name=filename,
                file_path=temp_file_path,
                file_size=file_size,
                error_message=str(e),
                confidence_threshold=confidence_threshold,
                batch_analysis_id=batch_id
            )
            
            return FileAnalysisResult(
                file_name=filename,
                file_path=filename,
                status=AnalysisStatus.FAILED,
                error_message=str(e)
            )
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    print(f"Warning: Failed to cleanup temp file {temp_file_path}: {e}")
    
    async def _finalize_batch_analysis(
        self, 
        batch_id: str, 
        file_results: List[FileAnalysisResult]
    ):
        """Finalize batch analysis with results"""
        try:
            batch_analysis = self.db.query(BatchAnalysis).filter(BatchAnalysis.id == batch_id).first()
            if not batch_analysis:
                return
            
            successful_files = len([r for r in file_results if r.status == AnalysisStatus.COMPLETED])
            failed_files = len([r for r in file_results if r.status == AnalysisStatus.FAILED])
            
            # Calculate summary statistics
            total_vulnerabilities = 0
            total_high_confidence = 0
            
            for result in file_results:
                if result.results and result.results.summary:
                    total_vulnerabilities += result.results.summary.total_vulnerabilities
                    total_high_confidence += result.results.summary.high_confidence
            
            # Determine batch status
            if failed_files == 0:
                status = BatchAnalysisStatus.COMPLETED
            elif successful_files > 0:
                status = BatchAnalysisStatus.PARTIAL_SUCCESS
            else:
                status = BatchAnalysisStatus.FAILED
            
            # Update batch analysis
            batch_analysis.successful_files = successful_files
            batch_analysis.failed_files = failed_files
            batch_analysis.total_vulnerabilities = total_vulnerabilities
            batch_analysis.total_high_confidence = total_high_confidence
            batch_analysis.status = status
            batch_analysis.completed_at = datetime.now()
            
            if batch_analysis.started_at:
                processing_time = (batch_analysis.completed_at - batch_analysis.started_at).total_seconds()
                batch_analysis.processing_time_seconds = processing_time
            
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            print(f"Error finalizing batch analysis {batch_id}: {e}")
    
    async def _mark_batch_failed(self, batch_id: str, error_message: str):
        """Mark batch analysis as failed"""
        try:
            batch_analysis = self.db.query(BatchAnalysis).filter(BatchAnalysis.id == batch_id).first()
            if batch_analysis:
                batch_analysis.status = BatchAnalysisStatus.FAILED
                batch_analysis.error_message = error_message
                batch_analysis.completed_at = datetime.now()
                self.db.commit()
        except Exception as e:
            self.db.rollback()
            print(f"Error marking batch failed {batch_id}: {e}")
    
    async def get_batch_analysis(self, batch_id: str) -> Optional[BatchAnalysisResult]:
        """Get batch analysis results"""
        try:
            batch_analysis = self.db.query(BatchAnalysis).filter(BatchAnalysis.id == batch_id).first()
            if not batch_analysis:
                return None
            
            # Get individual file analyses
            file_analyses = self.db.query(Analysis).filter(Analysis.batch_analysis_id == batch_id).all()
            
            file_results = []
            most_vulnerable_files = []
            
            for analysis in file_analyses:
                result_data = None
                if analysis.results_json:
                    try:
                        result_data = json.loads(analysis.results_json)
                    except json.JSONDecodeError:
                        pass
                
                file_result = FileAnalysisResult(
                    file_name=analysis.file_name,
                    file_path=analysis.file_name,
                    status=analysis.status,
                    results=result_data,
                    error_message=analysis.error_message,
                    processing_time_seconds=analysis.processing_time_seconds
                )
                file_results.append(file_result)
                
                # Track most vulnerable files
                if analysis.vulnerabilities_count and analysis.vulnerabilities_count > 0:
                    most_vulnerable_files.append(analysis.file_name)
            
            # Sort most vulnerable files by vulnerability count
            most_vulnerable_files = sorted(
                most_vulnerable_files,
                key=lambda f: next(
                    (a.vulnerabilities_count for a in file_analyses if a.file_name == f), 0
                ),
                reverse=True
            )[:5]  # Top 5
            
            return BatchAnalysisResult(
                batch_id=batch_id,
                name=batch_analysis.name,
                status=batch_analysis.status,
                total_files=batch_analysis.total_files,
                successful_files=batch_analysis.successful_files,
                failed_files=batch_analysis.failed_files,
                processing_time_seconds=batch_analysis.processing_time_seconds,
                file_results=file_results,
                total_vulnerabilities=batch_analysis.total_vulnerabilities,
                total_high_confidence=batch_analysis.total_high_confidence,
                most_vulnerable_files=most_vulnerable_files
            )
            
        except Exception as e:
            raise Exception(f"Failed to get batch analysis: {str(e)}")
    
    async def get_batch_analyses(self, limit: int = 10, offset: int = 0) -> BatchAnalysisListResponse:
        """Get list of batch analyses with pagination"""
        try:
            # Get total count
            total = self.db.query(BatchAnalysis).count()
            
            # Get batch analyses with pagination
            batch_analyses = (
                self.db.query(BatchAnalysis)
                .order_by(desc(BatchAnalysis.created_at))
                .offset(offset)
                .limit(limit)
                .all()
            )
            
            # Convert to response format
            batch_items = []
            for batch in batch_analyses:
                item = BatchAnalysisListItem(
                    id=batch.id,
                    name=batch.name,
                    timestamp=batch.created_at,
                    total_files=batch.total_files,
                    successful_files=batch.successful_files,
                    failed_files=batch.failed_files,
                    status=batch.status,
                    total_vulnerabilities=batch.total_vulnerabilities,
                    processing_time_seconds=batch.processing_time_seconds
                )
                batch_items.append(item)
            
            return BatchAnalysisListResponse(
                batch_analyses=batch_items,
                total=total,
                limit=limit,
                offset=offset
            )
            
        except Exception as e:
            raise Exception(f"Failed to get batch analyses: {str(e)}")
    
    async def delete_batch_analysis(self, batch_id: str) -> bool:
        """Delete batch analysis and all related individual analyses"""
        try:
            # Delete individual analyses first
            self.db.query(Analysis).filter(Analysis.batch_analysis_id == batch_id).delete()
            
            # Delete batch analysis
            batch_analysis = self.db.query(BatchAnalysis).filter(BatchAnalysis.id == batch_id).first()
            if not batch_analysis:
                return False
            
            self.db.delete(batch_analysis)
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to delete batch analysis: {str(e)}")
    async def get_detailed_batch_analyses(self, limit: int = 10, offset: int = 0) -> 'BatchAnalysisDetailedListResponse':
        """Get detailed list of batch analyses with full information including file results"""
        try:
            from app.schemas.batch_analysis import BatchAnalysisDetailedListResponse, BatchAnalysisDetailedItem
            
            # Get total count
            total = self.db.query(BatchAnalysis).count()
            
            # Get batch analyses with pagination
            batch_analyses = (
                self.db.query(BatchAnalysis)
                .order_by(desc(BatchAnalysis.created_at))
                .offset(offset)
                .limit(limit)
                .all()
            )
            
            # Convert to detailed response format
            detailed_items = []
            for batch in batch_analyses:
                # Get file analyses for this batch
                file_analyses = (
                    self.db.query(Analysis)
                    .filter(Analysis.batch_analysis_id == batch.id)
                    .all()
                )
                
                # Convert file analyses to FileAnalysisResult
                file_results = []
                most_vulnerable_files = []
                
                for analysis in file_analyses:
                    # Parse results JSON
                    results_data = None
                    if analysis.results_json:
                        try:
                            results_data = json.loads(analysis.results_json)
                        except:
                            results_data = None
                    
                    # Create FileAnalysisResult
                    file_result = FileAnalysisResult(
                        file_name=analysis.file_name,
                        file_path=analysis.file_path or "",
                        status=analysis.status,
                        results=results_data,
                        error_message=analysis.error_message,
                        processing_time_seconds=analysis.processing_time_seconds
                    )
                    file_results.append(file_result)
                    
                    # Track most vulnerable files
                    if analysis.vulnerabilities_count and analysis.vulnerabilities_count > 0:
                        most_vulnerable_files.append(analysis.file_name)
                
                # Sort most vulnerable files by vulnerability count
                most_vulnerable_files.sort(key=lambda x: next(
                    (a.vulnerabilities_count for a in file_analyses if a.file_name == x), 0
                ), reverse=True)
                
                # Create detailed item
                detailed_item = BatchAnalysisDetailedItem(
                    id=batch.id,
                    name=batch.name,
                    timestamp=batch.created_at,
                    total_files=batch.total_files,
                    successful_files=batch.successful_files,
                    failed_files=batch.failed_files,
                    status=batch.status,
                    total_vulnerabilities=batch.total_vulnerabilities,
                    total_high_confidence=batch.total_high_confidence or 0,
                    processing_time_seconds=batch.processing_time_seconds,
                    confidence_threshold=batch.confidence_threshold,
                    source_type=batch.source_type,
                    source_info=batch.source_info,
                    error_message=batch.error_message,
                    started_at=batch.started_at,
                    completed_at=batch.completed_at,
                    file_results=file_results,
                    most_vulnerable_files=most_vulnerable_files[:5]  # Top 5 most vulnerable files
                )
                detailed_items.append(detailed_item)
            
            return BatchAnalysisDetailedListResponse(
                batch_analyses=detailed_items,
                total=total,
                limit=limit,
                offset=offset
            )
            
        except Exception as e:
            raise Exception(f"Failed to get detailed batch analyses: {str(e)}")
