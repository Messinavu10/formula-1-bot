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
from collections import defaultdict

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

app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Initialize MCP pipeline
mcp_pipeline = None

# Session management - store conversation history per session
session_conversations = defaultdict(list)

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
        session_id = data.get("session_id", "default")
        
        if not user_message:
            return {"error": "Message cannot be empty"}
        
        if not mcp_pipeline:
            return {"error": "F1 Assistant is not ready. Please try again."}
        
        # Get conversation history for this session
        conversation_history = session_conversations[session_id]
        
        # Update the reasoning engine's context manager with the session history
        mcp_pipeline.reasoning_engine.context_manager.conversation_history = conversation_history
        
        # Add debug logging
        print(f"🔍 Session {session_id}: Restored {len(conversation_history)} conversation entries")
        
        # Get response from reasoning engine
        result = mcp_pipeline.test_reasoning_engine(user_message)
        
        # Handle the response (it might be a tuple now)
        if isinstance(result, tuple):
            response, visualization_data = result
        else:
            # Backward compatibility for when test_reasoning_engine returns just a string
            response = result
            visualization_data = None
        
        # Update session history with the new exchange
        session_conversations[session_id] = mcp_pipeline.reasoning_engine.context_manager.conversation_history
        
        # Add debug logging
        print(f"🔍 Session {session_id}: Updated to {len(session_conversations[session_id])} conversation entries")
        
        # Add debug logging for visualization data
        if visualization_data:
            print(f"🔍 Visualization data: {visualization_data}")
            print(f"🔍 Visualization filename: {visualization_data.get('filename', 'N/A')}")
            print(f"🔍 Visualization success: {visualization_data.get('success', False)}")
        
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
    # Look in the visualizations subdirectory
    file_path = Path("artifacts/visualizations") / filename
    if file_path.exists():
        return FileResponse(file_path)
    else:
        # Add debug logging to see what path is being checked
        print(f"🔍 Looking for visualization file: {file_path}")
        print(f"�� File exists: {file_path.exists()}")
        return {"error": "Visualization not found"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)