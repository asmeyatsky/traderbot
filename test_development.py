#!/usr/bin/env python3
"""
Test script to verify the application can run in development mode
"""
import asyncio
import os
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import requests
from src.presentation.api.main import app
import uvicorn
from threading import Thread

def run_server():
    """Run the FastAPI server in a separate thread for testing."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    print("Starting test server...")
    
    # Start server in a separate thread
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    # Test the root endpoint
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        print(f"Root endpoint response: {response.status_code}")
        print(f"Response content: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to server - might be an issue with the setup")
    except Exception as e:
        print(f"Error testing endpoint: {e}")
    
    # Test the health endpoint
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print(f"Health endpoint response: {response.status_code}")
        print(f"Response content: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to health endpoint")
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
    
    print("Development mode test completed.")
    print("Note: Server would continue running in production use.")