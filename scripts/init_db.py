#!/usr/bin/env python3
"""
Database initialization script
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import engine, Base
from app.config import settings

def init_database():
    """Initialize the database with all tables"""
    print("🗄️  Initializing database...")
    print(f"📍 Database URL: {settings.DATABASE_URL}")
    
    try:
        # Import all models to ensure they're registered
        from app.models.analysis import Analysis
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        print("✅ Database initialized successfully!")
        print("📊 Tables created:")
        for table_name in Base.metadata.tables.keys():
            print(f"   - {table_name}")
            
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()