import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv('.env')

print("Environment variables:")
print(f"POLYGON_API_KEY: {os.getenv('POLYGON_API_KEY')}")
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")

# Define a simple settings model
class TestSettings(BaseModel):
    POLYGON_API_KEY: str = Field(..., min_length=1)
    DATABASE_URL: str = Field(..., min_length=1)

# Try to create the settings instance
try:
    test_settings = TestSettings()
    print(f"Test settings created successfully: {test_settings.DATABASE_URL}")
except Exception as e:
    print(f"Error creating test settings: {e}")