#!/usr/bin/env python3
"""
Migration script to add app_settings table
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import engine, Base
from app.models.app_settings import AppSettings
from app.services.settings_service import SettingsService
from app.config import settings

def run_migration():
    """Run the migration to add app_settings table"""
    print("🔄 Running app_settings migration...")
    print(f"📍 Database URL: {settings.DATABASE_URL}")
    
    try:
        # Create the app_settings table
        AppSettings.__table__.create(bind=engine, checkfirst=True)
        print("✅ app_settings table created")
        
        # Initialize default settings
        print("⚙️  Initializing default settings...")
        settings_service = SettingsService()
        
        # Run the async function
        import asyncio
        asyncio.run(settings_service.initialize_default_settings())
        
        print("✅ Default settings initialized")
        print("🏁 Migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_migration()
