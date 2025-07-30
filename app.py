from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import asyncio
from typing import Dict, Any
import logging
from pathlib import Path

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("_client").setLevel(logging.WARNING)

# Import your MCP pipeline
from src.formula_one.pipeline.mcp_pipeline import MCPTrainingPipeline

app = FastAPI(
    title="F1 Racing Assistant",
    description="Your AI-powered Formula 1 racing assistant",
    version="2.0.0"
)

# Mount templates and static files
templates = Jinja2Templates(directory="templates")

# Create artifacts directory for visualizations
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

# Mount static files for serving visualizations
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")

# Initialize MCP pipeline
mcp_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize the MCP pipeline on startup"""
    global mcp_pipeline
    try:
        mcp_pipeline = MCPTrainingPipeline()
        # Start the server but don't enter interactive mode
        mcp_pipeline.start_server()
        # Wait for server to be ready
        if not mcp_pipeline.client.wait_for_server(max_wait=30):
            raise Exception("MCP Server failed to start")
        print("🚀 F1 Racing Assistant is ready!")
    except Exception as e:
        print(f"❌ Failed to initialize MCP pipeline: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global mcp_pipeline
    if mcp_pipeline and hasattr(mcp_pipeline, 'server'):
        mcp_pipeline.server.stop_server()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Handle chat requests"""
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return {"error": "Message cannot be empty"}
        
        if not mcp_pipeline:
            return {"error": "F1 Assistant is not ready. Please try again."}
        
        # Get response from reasoning engine
        response = mcp_pipeline.test_reasoning_engine(user_message)
        
        # Check if response contains visualization data
        visualization_data = None
        if hasattr(mcp_pipeline, 'last_visualization') and mcp_pipeline.last_visualization:
            visualization_data = mcp_pipeline.last_visualization
            mcp_pipeline.last_visualization = None  # Clear after sending
        
        return {
            "response": response,
            "visualization": visualization_data,
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logging.error(f"Error processing chat request: {e}")
        return {"error": "Sorry, I encountered an error. Please try again."}

@app.get("/api/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve visualization files"""
    try:
        viz_path = Path("artifacts/visualizations") / filename
        if viz_path.exists():
            return FileResponse(
                path=str(viz_path),
                media_type="text/html",
                # Remove filename parameter to prevent download
                headers={
                    "Content-Disposition": "inline",  # Display in browser, don't download
                    "X-Frame-Options": "SAMEORIGIN"  # Allow iframe embedding
                }
            )
        else:
            return {"error": "Visualization not found"}
    except Exception as e:
        logging.error(f"Error serving visualization {filename}: {e}")
        return {"error": "Error serving visualization"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mcp_ready": mcp_pipeline is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )