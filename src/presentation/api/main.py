"""
Main application module for the AI Trading Platform.

This follows the clean architecture pattern with dependency injection.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.infrastructure.config.settings import settings

# Initialize the application
app = FastAPI(
    title="AI Trading Platform API",
    description="API for the AI-powered autonomous trading platform",
    version="0.1.0"
)

# Parse allowed origins from settings
allowed_origins = settings.ALLOWED_ORIGINS.split(",")

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Trading Platform API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Trading Platform API"}

# Include API routes (will be added as we develop features)
# from src.presentation.api.routers import market_data, trading, user, etc.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)