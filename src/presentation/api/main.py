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

# Initialize the application
app = FastAPI(
    title="AI Trading Platform API",
    description="API for the AI-powered autonomous trading platform",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, configure this properly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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