from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
import asyncio
import json
import threading
import time
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.mcp_config_entity import MCPConfig
from src.formula_one.entity.config_entity import DatabaseConfig
from typing import Optional 

class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCallResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None  # Make error optional

class ToolInfo(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]
    examples: List[Dict[str, Any]] = []

class FastAPIServer(BaseComponent):
    """FastAPI server for MCP tools"""
    
    def __init__(self, config, db_config: DatabaseConfig, tools: Dict[str, Any]):
        super().__init__(config, db_config)
        self.tools = tools
        self.app = FastAPI(
            title="F1 MCP Server",
            version="2.0.0",
            description="Enhanced MCP Server for Formula 1 data analysis"
        )
        self.server_running = False
        self.server_thread = None
        
        # 🔐 Initialize security first
        self.security = HTTPBearer()
        self._setup_security()
        
        # Then setup routes
        self._setup_routes()
    
    def _setup_security(self):
        """Setup security middleware and dependencies"""
        
        async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Verify API token"""
            api_token = os.getenv("API_TOKEN")
            
            # If no API_TOKEN is set, allow all requests (development mode)
            if not api_token:
                self.logger.warning("⚠️ No API_TOKEN set - running in development mode (no authentication)")
                return credentials.credentials
            
            if credentials.credentials != api_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API token"
                )
            return credentials.credentials
        
        # Store the verification function for use in routes
        self.verify_token = verify_token
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Enhanced F1 MCP Server is running!",
                "version": "1.0.0",
                "endpoints": {
                    "tools": "/tools",
                    "call_tool": "/call_tool (🔐 Protected)",
                    "health": "/health",
                    "docs": "/docs",
                    "schema": "/schema (🔐 Protected)"
                },
                "tool_categories": {
                    "core_identification": ["get_meeting_key", "get_session_key"],
                    "performance_analysis": ["get_fastest_lap", "get_driver_performance", "get_team_performance"],
                    "comparison": ["compare_drivers", "compare_teams"],
                    "results": ["get_race_results", "get_qualifying_results"],
                    "strategy": ["get_pit_stop_analysis", "get_tire_strategy"],
                    "incident_analysis": ["investigate_incident"],
                    "position_analysis": ["get_position_progression"],
                    "sector_analysis": ["get_sector_analysis"],
                    "utility": ["explore_schema", "get_session_info"]
                }
            }
        
        @self.app.get("/tools")
        async def list_tools():
            tools_info = []
            for tool_name, tool in self.tools.items():
                tools_info.append(tool.get_tool_info())
            return {"tools": tools_info}
        
        # �� PROTECTED ENDPOINT - Requires API token
        @self.app.post("/call_tool")
        async def call_tool(request: ToolCallRequest, token: str = Depends(self.verify_token)):
            if request.name not in self.tools:
                raise HTTPException(status_code=404, detail=f"Tool {request.name} not found")
            
            try:
                tool = self.tools[request.name]
                result = await tool.execute(request.arguments)
                
                return ToolCallResponse(
                    success=result.success,
                    data=result.data,
                    error=result.error if result.error else None  # Handle None explicitly
                )
            except Exception as e:
                return ToolCallResponse(
                    success=False,
                    data={},
                    error=str(e)
                )
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "server": "enhanced-f1-mcp-server",
                "version": "2.0.0",
                "database_connected": True,
                "tool_count": len(self.tools),
                "security_enabled": bool(os.getenv("API_TOKEN"))
            }
        
        # �� PROTECTED ENDPOINT - Requires API token
        @self.app.get("/schema")
        async def get_schema(token: str = Depends(self.verify_token)):
            return {"schema": self.db_utils.get_schema_info()}
        
        @self.app.get("/tool_status/{tool_name}")
        async def get_tool_status(tool_name: str):
            """Get status of a specific tool"""
            if tool_name not in self.tools:
                raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
            
            tool = self.tools[tool_name]
            return {
                "tool_name": tool_name,
                "status": "available",
                "category": getattr(tool, 'category', 'unknown'),
                "description": tool.__doc__ or ""
            }
    
    def run_server_in_thread(self, host: str = None, port: int = None):
        """Run the server in a separate thread"""
        def run():
            uvicorn.run(self.app, host=host or "localhost", port=port or 8000, log_level="info")
        
        if self.server_running:
            self.logger.warning("⚠️ Server is already running!")
            return
        
        self.server_thread = threading.Thread(target=run, daemon=True)
        self.server_thread.start()
        self.server_running = True
        
        # Wait a moment for server to start
        time.sleep(2)
        self.logger.info(f"✅ Enhanced F1 MCP HTTP Server started on {host or 'localhost'}:{port or 8000}")
        self.logger.info(f"�� API Documentation: http://{host or 'localhost'}:{port or 8000}/docs")
        self.logger.info(f"�� Health Check: http://{host or 'localhost'}:{port or 8000}/health")
        self.logger.info(f"🛠️ Available Tools: http://{host or 'localhost'}:{port or 8000}/tools")
    
    def stop_server(self):
        """Stop the server"""
        if self.server_running:
            self.server_running = False
            self.logger.info("🛑 Server stopped")
        else:
            self.logger.warning("⚠️ Server is not running")
    
    def check_server_status(self):
        """Check if the server is running"""
        if self.server_running:
            self.logger.info("✅ Server is running")
            try:
                import requests
                response = requests.get("http://localhost:8000/health")
                if response.status_code == 200:
                    self.logger.info("✅ Server is responding to requests")
                    health_data = response.json()
                    self.logger.info(f"✅ Tool count: {health_data.get('tool_count', 'Unknown')}")
                else:
                    self.logger.warning("⚠️ Server is running but not responding properly")
            except Exception as e:
                self.logger.warning(f"⚠️ Server is running but not accessible: {e}")
        else:
            self.logger.error("❌ Server is not running")