#!/usr/bin/env python3
"""
Test script to verify environment variable loading
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv('.env')

print("Environment variables after loading:")
print(f"POLYGON_API_KEY: {os.getenv('POLYGON_API_KEY', 'NOT SET')}")
print(f"ALPHA_VANTAGE_API_KEY: {os.getenv('ALPHA_VANTAGE_API_KEY', 'NOT SET')}")
print(f"DATABASE_URL: {os.getenv('DATABASE_URL', 'NOT SET')}")
print(f"JWT_SECRET_KEY: {os.getenv('JWT_SECRET_KEY', 'NOT SET')}")

# Now try to import settings - only the module, not the object
try:
    import src.infrastructure.config.settings as settings_module
    print("Module imported successfully")

    # Now try to access the settings object
    settings = settings_module.settings
    print(f"Settings accessed successfully. DATABASE_URL: {settings.DATABASE_URL}")
except Exception as e:
    print(f"Error accessing settings: {e}")
    import traceback
    traceback.print_exc()