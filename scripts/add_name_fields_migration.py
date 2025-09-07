#!/usr/bin/env python3
"""
Migration script to add 'name' fields to analyses and batch_analyses tables
Run this script to update existing databases with the new name fields.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Add name columns to existing tables"""
    engine = create_engine(settings.DATABASE_URL)
    
    try:
        with engine.connect() as conn:
            logger.info("Starting database migration to add name fields...")
            
            # Check if analyses table exists and add name column if it doesn't exist
            try:
                # Try to add name column to analyses table
                conn.execute(text("ALTER TABLE analyses ADD COLUMN name VARCHAR"))
                logger.info("✅ Added 'name' column to analyses table")
            except Exception as e:
                if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("ℹ️  'name' column already exists in analyses table")
                else:
                    logger.error(f"❌ Error adding 'name' column to analyses table: {e}")
                    
            # Check if batch_analyses table exists and add name column if it doesn't exist
            try:
                # Try to add name column to batch_analyses table
                conn.execute(text("ALTER TABLE batch_analyses ADD COLUMN name VARCHAR"))
                logger.info("✅ Added 'name' column to batch_analyses table")
            except Exception as e:
                if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                    logger.info("ℹ️  'name' column already exists in batch_analyses table")
                else:
                    logger.error(f"❌ Error adding 'name' column to batch_analyses table: {e}")
            
            # Commit the changes
            conn.commit()
            logger.info("✅ Database migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    run_migration() 