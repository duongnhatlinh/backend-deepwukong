"""
Custom exceptions and exception handlers
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback

logger = logging.getLogger(__name__)

# Custom Exceptions
class DeepWukongBaseException(Exception):
    """Base exception for DeepWukong application"""
    pass

class ModelLoadError(DeepWukongBaseException):
    """Raised when AI model fails to load"""
    pass

class AnalysisError(DeepWukongBaseException):
    """Raised when analysis fails"""
    pass

class ValidationError(DeepWukongBaseException):
    """Raised when input validation fails"""
    pass

class FileProcessingError(DeepWukongBaseException):
    """Raised when file processing fails"""
    pass

# Exception Handlers
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation exceptions"""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation Error",
            "details": exc.errors(),
            "status_code": 422
        }
    )

async def deepwukong_exception_handler(request: Request, exc: DeepWukongBaseException):
    """Handle custom DeepWukong exceptions"""
    logger.error(f"DeepWukong error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "type": exc.__class__.__name__,
            "status_code": 500
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "details": str(exc) if logger.level == logging.DEBUG else None,
            "status_code": 500
        }
    )

def setup_exception_handlers(app: FastAPI):
    """Setup exception handlers for the FastAPI app"""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(DeepWukongBaseException, deepwukong_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)