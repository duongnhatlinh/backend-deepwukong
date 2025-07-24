#!/usr/bin/env python3
"""
DeepWukong Backend Server
Entry point for running the FastAPI application
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from app.main import app
from app.config import settings

def main():
    """Main function to run the server"""
    print(f"🚀 Starting {settings.APP_NAME} v{settings.VERSION}")
    print(f"📝 Debug mode: {settings.DEBUG}")
    print(f"🌐 Server will be available at: http://{settings.HOST}:{settings.PORT}")
    print(f"📚 API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"📋 Alternative docs: http://{settings.HOST}:{settings.PORT}/redoc")
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
        access_log=True
    )

if __name__ == "__main__":
    main()