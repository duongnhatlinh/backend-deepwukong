"""
DeepWuKong AI Model Service
Wrapper around the original DeepWuKong implementation
"""

import asyncio
import json
import os
import tempfile
import time
import sys
from typing import Dict, List, Optional
from pathlib import Path

from app.config import settings
from app.core.exceptions import ModelLoadError, AnalysisError

# Add deepwukong to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../deepwukong'))

try:
    from src.enhanced_detect import VulnerabilityDetector
except ImportError as e:
    print(f"Warning: Could not import VulnerabilityDetector: {e}")
    VulnerabilityDetector = None

class DeepWuKongService:
    def __init__(self):
        self.detector: Optional[VulnerabilityDetector] = None
        self.model_loaded = False
        self.model_info = {}
        
    async def initialize(self) -> bool:
        """Initialize the DeepWuKong model"""
        try:
            print("🤖 Loading DeepWuKong model...")
            
            # Check if DeepWuKong is available
            if VulnerabilityDetector is None:
                raise ModelLoadError("DeepWuKong source code not available")
            
            # Check if model file exists
            if not os.path.exists(settings.DEEPWUKONG_MODEL_PATH):
                print(f"⚠️  Model file not found: {settings.DEEPWUKONG_MODEL_PATH}")
                print("📝 Using mock mode for development")
                self.model_loaded = True
                self.model_info = {
                    "version": "DeepWuKong-v1.0-MOCK",
                    "architecture": "Mock Mode for Development", 
                    "load_time": 0.1,
                    "model_path": "MOCK",
                    "status": "mock"
                }
                return True
            
            # Load model in executor to avoid blocking
            loop = asyncio.get_event_loop()
            start_time = time.time()
            
            self.detector = await loop.run_in_executor(
                None, 
                VulnerabilityDetector,
                settings.DEEPWUKONG_MODEL_PATH
            )
            
            load_time = time.time() - start_time
            self.model_loaded = True
            
            # Store model info
            self.model_info = {
                "version": "DeepWuKong-v1.0",
                "architecture": "GCN + RNN Classifier", 
                "load_time": round(load_time, 2),
                "model_path": settings.DEEPWUKONG_MODEL_PATH,
                "status": "ready"
            }
            
            print(f"✅ Model loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            # Fall back to mock mode
            print("📝 Falling back to mock mode")
            self.model_loaded = True
            self.model_info = {
                "version": "DeepWuKong-v1.0-MOCK",
                "architecture": "Mock Mode (Model Load Failed)", 
                "load_time": 0.1,
                "model_path": "MOCK",
                "status": "mock",
                "error": str(e)
            }
            return True
    
    async def analyze_file(self, file_path: str, options: Dict = None) -> Dict:
        """Analyze a single file for vulnerabilities"""
        if not self.model_loaded:
            raise AnalysisError("Model not loaded")
        
        try:
            start_time = time.time()
            
            # Check if running in mock mode
            if self.model_info.get("status") == "mock":
                # Return mock results for development
                results = self._generate_mock_results(file_path)
            else:
                # Run real analysis in executor to avoid blocking
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    self.detector.detect_files,
                    file_path
                )
            
            processing_time = time.time() - start_time
            
            # Format results for API response
            formatted_results = self._format_results(results, file_path, processing_time)
            
            return formatted_results
            
        except Exception as e:
            raise AnalysisError(f"Analysis failed: {str(e)}")
    
    def _generate_mock_results(self, file_path: str) -> List[Dict]:
        """Generate mock results for development/testing"""
        # Read file to get realistic line numbers
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                total_lines = len(lines)
        except:
            total_lines = 100
        
        # Generate some mock vulnerabilities
        mock_results = []
        
        # Mock buffer overflow
        if total_lines > 10:
            mock_results.append({
                "file_path": file_path,
                "line_number": min(15, total_lines - 5),
                "api_type": "array",
                "is_vulnerable": True,
                "confidence": 0.85,
                "vulnerability_probability": 0.85,
                "prediction_class": 1
            })
        
        # Mock unsafe function call
        if total_lines > 20:
            mock_results.append({
                "file_path": file_path,
                "line_number": min(25, total_lines - 2),
                "api_type": "call",
                "is_vulnerable": True,
                "confidence": 0.72,
                "vulnerability_probability": 0.72,
                "prediction_class": 1
            })
        
        # Mock potential null pointer
        if total_lines > 5:
            mock_results.append({
                "file_path": file_path,
                "line_number": min(8, total_lines - 1),
                "api_type": "ptr",
                "is_vulnerable": True,
                "confidence": 0.65,
                "vulnerability_probability": 0.65,
                "prediction_class": 1
            })
        
        return mock_results
    
    def _format_results(self, raw_results: List[Dict], file_path: str, processing_time: float) -> Dict:
        """Format DeepWuKong results for API response"""
        
        vulnerabilities = []
        
        for result in raw_results:
            vulnerability = {
                "line_number": result.get("line_number", 0),
                "type": self._map_vulnerability_type(result.get("api_type", "unknown")),
                "confidence": round(result.get("confidence", 0.0), 3),
                "severity": self._map_severity(result.get("confidence", 0.0)),
                "description": self._get_description(result.get("api_type"), result.get("confidence", 0.0)),
                "code_snippet": self._format_code_snippet(result.get("code_snippet")),
                "api_type": result.get("api_type", "unknown"),
                "recommendation": self._get_recommendation(result.get("api_type"))
            }
            vulnerabilities.append(vulnerability)
        
        # Sort by confidence (highest first)
        vulnerabilities.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Create summary
        summary = {
            "total_vulnerabilities": len(vulnerabilities),
            "high_confidence": len([v for v in vulnerabilities if v["confidence"] >= 0.8]),
            "medium_confidence": len([v for v in vulnerabilities if 0.6 <= v["confidence"] < 0.8]),
            "low_confidence": len([v for v in vulnerabilities if v["confidence"] < 0.6]),
            "processing_time": f"{processing_time:.2f}s",
            "severity_counts": {
                "critical": len([v for v in vulnerabilities if v["severity"] == "critical"]),
                "high": len([v for v in vulnerabilities if v["severity"] == "high"]),
                "medium": len([v for v in vulnerabilities if v["severity"] == "medium"]),
                "low": len([v for v in vulnerabilities if v["severity"] == "low"])
            }
        }
        
        return {
            "file_analyzed": os.path.basename(file_path),
            "vulnerabilities": vulnerabilities,
            "summary": summary,
            "processing_time": f"{processing_time:.2f}s",
            "model_version": self.model_info.get("version", "unknown")
        }
    
    def _map_vulnerability_type(self, api_type: str) -> str:
        """Map API type to vulnerability type"""
        mapping = {
            "call": "unsafe_function_call",
            "array": "buffer_overflow", 
            "ptr": "null_pointer_dereference",
            "arith": "integer_overflow"
        }
        return mapping.get(api_type, "unknown_vulnerability")
    
    def _map_severity(self, confidence: float) -> str:
        """Map confidence to severity level"""
        if confidence >= 0.9:
            return "critical"
        elif confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_description(self, api_type: str, confidence: float) -> str:
        """Get description based on API type and confidence"""
        base_descriptions = {
            "call": "Potentially unsafe function call detected",
            "array": "Possible buffer overflow vulnerability",
            "ptr": "Potential null pointer dereference",
            "arith": "Possible integer overflow condition"
        }
        
        base = base_descriptions.get(api_type, "Potential vulnerability detected")
        
        if confidence >= 0.9:
            return f"{base} with very high confidence"
        elif confidence >= 0.8:
            return f"{base} with high confidence"
        elif confidence >= 0.6:
            return f"{base} with moderate confidence"
        else:
            return f"{base} with low confidence"
    
    def _get_recommendation(self, api_type: str) -> str:
        """Get recommendation based on API type"""
        recommendations = {
            "call": "Use safer alternatives or validate inputs properly. Consider using safe string functions.",
            "array": "Add bounds checking before array access. Validate array indices.",
            "ptr": "Check for null pointers before dereferencing. Initialize pointers properly.", 
            "arith": "Validate arithmetic operations to prevent overflow. Use appropriate data types."
        }
        return recommendations.get(api_type, "Review code for potential security issues")
    
    def _format_code_snippet(self, snippet_data) -> Optional[List[Dict]]:
        """Convert code snippet dictionary to list of CodeSnippetSchema objects"""
        if not snippet_data or not isinstance(snippet_data, dict):
            return None
        
        try:
            snippet_type = snippet_data.get('type', 'unknown')
            
            if snippet_type == 'context':
                lines = snippet_data.get('lines', [])
                start_line_num = snippet_data.get('start_line_num', 1)
                highlight_line_index = snippet_data.get('highlight_line_index', -1)
                
                if not lines:
                    return None
                
                code_snippets = []
                for i, line in enumerate(lines):
                    line_num = start_line_num + i
                    is_highlighted = (i == highlight_line_index)
                    
                    code_snippets.append({
                        "lineNumber": line_num,
                        "content": line,
                        "isHighlighted": is_highlighted
                    })
                
                return code_snippets
            
            elif snippet_type == 'error':
                return [{
                    "lineNumber": 0,
                    "content": f"Error: {snippet_data.get('message', 'Unknown error')}",
                    "isHighlighted": False
                }]
            
            else:
                # Fallback: try to extract any text content
                if 'lines' in snippet_data:
                    code_snippets = []
                    for i, line in enumerate(snippet_data['lines']):
                        code_snippets.append({
                            "lineNumber": i + 1,
                            "content": line,
                            "isHighlighted": False
                        })
                    return code_snippets
                
                return [{
                    "lineNumber": 0,
                    "content": str(snippet_data),
                    "isHighlighted": False
                }]
                
        except Exception as e:
            return [{
                "lineNumber": 0,
                "content": f"Error formatting snippet: {str(e)}",
                "isHighlighted": False
            }]
    
    def get_status(self) -> Dict:
        """Get current model status"""
        return {
            "model_loaded": self.model_loaded,
            "model_info": self.model_info,
            "status": "ready" if self.model_loaded else "not_loaded"
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.detector:
            # Cleanup if needed
            self.detector = None
        self.model_loaded = False
        print("🧹 DeepWuKong service cleaned up")