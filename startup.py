#!/usr/bin/env python3
"""
Application startup script for Cloud Run
Respects the PORT environment variable as required by Cloud Run
"""

import os
import uvicorn
from src.presentation.api.main import app

if __name__ == "__main__":
    # Use PORT environment variable from Cloud Run, default to 8000 for local dev
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )