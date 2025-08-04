from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio
from typing import Dict, Any, Optional
import logging
from pathlib import Path
from collections import defaultdict

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("_client").setLevel(logging.WARNING)

# Import your MCP pipeline and session manager
from src.formula_one.pipeline.mcp_pipeline import MCPTrainingPipeline
from src.formula_one.utils.session_manager import session_manager

app = FastAPI(
    title="F1 Racing Assistant",
    description="Your AI-powered Formula 1 racing assistant",
    version="2.0.0"
)

# Add CORS middleware for multi-user support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount templates and static files
templates = Jinja2Templates(directory="templates")

# Create artifacts directory for visualizations
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

# Mount static files for serving visualizations
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")

# Only mount assets if the directory exists
if Path("assets").exists():
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Initialize MCP pipeline
mcp_pipeline = None

# Global MCP pipeline instance
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
    """Handle chat requests with session management"""
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        
        if not user_message:
            return {"error": "Message cannot be empty"}
        
        if not mcp_pipeline:
            return {"error": "F1 Assistant is not ready. Please try again."}
        
        # Create session if it doesn't exist
        if not session_id or not session_manager.validate_session(session_id):
            session_id = session_manager.create_session(user_id)
        
        # Update session activity
        session_manager.update_session_activity(session_id)
        
        # Get conversation history for this session
        conversation_history = session_manager.get_conversation_history(session_id)
        
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
        
        # Add conversation entry to session
        session_manager.add_conversation_entry(
            session_id, 
            user_message, 
            response, 
            {"visualization": visualization_data}
        )
        
        # Add debug logging
        print(f"🔍 Session {session_id}: Updated conversation history")
        
        # Add debug logging for visualization data
        if visualization_data:
            print(f"🔍 Visualization data: {visualization_data}")
            print(f"🔍 Visualization filename: {visualization_data.get('filename', 'N/A')}")
            print(f"🔍 Visualization success: {visualization_data.get('success', False)}")
        
        return {
            "response": response,
            "visualization": visualization_data,
            "session_id": session_id,
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

@app.post("/api/session/create")
async def create_session(user_id: Optional[str] = None):
    """Create a new session"""
    try:
        session_id = session_manager.create_session(user_id)
        return {
            "session_id": session_id,
            "user_id": user_id,
            "status": "created"
        }
    except Exception as e:
        logging.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.get("/api/session/{session_id}/validate")
async def validate_session(session_id: str):
    """Validate if a session exists and is active"""
    is_valid = session_manager.validate_session(session_id)
    return {
        "session_id": session_id,
        "is_valid": is_valid
    }

@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """End a session"""
    try:
        session_manager.end_session(session_id)
        return {
            "session_id": session_id,
            "status": "ended"
        }
    except Exception as e:
        logging.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail="Failed to end session")

@app.get("/api/session/stats")
async def get_session_stats():
    """Get session statistics"""
    try:
        stats = session_manager.get_session_stats()
        return stats
    except Exception as e:
        logging.error(f"Error getting session stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session stats")

@app.get("/api/session/user/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all sessions for a user"""
    try:
        session_ids = session_manager.get_user_sessions(user_id)
        return {
            "user_id": user_id,
            "session_ids": session_ids,
            "count": len(session_ids)
        }
    except Exception as e:
        logging.error(f"Error getting user sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user sessions")

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)