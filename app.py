from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import json
import asyncio
from typing import Dict, Any
import logging

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

# Only mount templates - no static files needed
templates = Jinja2Templates(directory="templates")

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
        
        return {
            "response": response,
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logging.error(f"Error processing chat request: {e}")
        return {"error": "Sorry, I encountered an error. Please try again."}

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
        reload=True,
        log_level="info"
    )