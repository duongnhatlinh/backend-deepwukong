"""
FastAPI Application Main Module
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from app.config import settings
from app.database import init_db
from app.services.deepwukong_service import DeepWuKongService
from app.api import ai, analyze, health
from app.core.exceptions import setup_exception_handlers

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
deepwukong_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global deepwukong_service
    
    # Startup
    logger.info("🚀 Starting DeepWukong Backend...")
    
    try:
        # Initialize database
        logger.info("📊 Initializing database...")
        init_db()
        logger.info("✅ Database initialized")
        
        # Initialize DeepWukong service
        logger.info("🤖 Initializing DeepWukong service...")
        deepwukong_service = DeepWuKongService()
        await deepwukong_service.initialize()
        
        # Store in app state
        app.state.deepwukong = deepwukong_service
        
        logger.info("✅ DeepWukong Backend started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down DeepWukong Backend...")
    if deepwukong_service:
        await deepwukong_service.cleanup()
    logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="DeepWukong API",
    description="""
    ## DeepWukong Vulnerability Detection API
    
    This API provides automated vulnerability detection for C/C++ source code using the DeepWukong AI model.
    
    ### Features:
    - **AI-Powered Analysis**: Uses Deep Graph Neural Networks for vulnerability detection
    - **Multiple Vulnerability Types**: Detects buffer overflows, null pointer dereferences, integer overflows, and unsafe function calls
    - **Confidence Scoring**: Provides confidence levels for each detected vulnerability
    - **File Upload**: Supports direct file upload for analysis
    - **Analysis History**: Stores and retrieves analysis results
    
    ### Supported File Types:
    C/C++ source files: `.c`, `.cpp`, `.h`, `.hpp`, `.cc`, `.cxx`
    
    ### Model Information:
    Based on the research paper: "DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network" (TOSEM'21)
    """,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Static files (if storage directory exists)
if os.path.exists("storage"):
    app.mount("/static", StaticFiles(directory="storage"), name="static")

# Exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI Model"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "DeepWukong Vulnerability Detection API",
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/health",
        "ai_status": "/api/ai/status"
    }

@app.get("/api", tags=["Root"])
async def api_info():
    """API information endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "description": "AI-powered vulnerability detection for C/C++ code",
        "endpoints": {
            "health": "/api/health",
            "ai_status": "/api/ai/status",
            "ai_info": "/api/ai/info", 
            "analyze": "/api/analyze",
            "analyses": "/api/analyses"
        },
        "supported_files": settings.ALLOWED_EXTENSIONS,
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB
    }