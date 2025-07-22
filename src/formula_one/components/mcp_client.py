import requests
import json
from typing import Dict, Any, List, Optional
import time
import logging
import os

from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.config_entity import DatabaseConfig

class HTTPMCPClient(BaseComponent):
    """HTTP-based MCP Client to connect to the F1 MCP Server"""
    
    def __init__(self, config, db_config: DatabaseConfig, base_url: str = "http://localhost:8000"):
        super().__init__(config, db_config)
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 10  # 10 second timeout

        api_token = os.getenv("API_TOKEN")
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})
            self.logger.info("🔐 API token configured for authenticated requests")
        else:
            self.logger.warning("⚠️ No API_TOKEN found - requests may fail if server requires authentication")
    
    def call_tool(self, name: str, arguments: Dict[str, Any], retries=2) -> Dict[str, Any]:
        """Call a tool on the server with retry logic"""
        try:
            payload = {"name": name, "arguments": arguments}
            for attempt in range(retries + 1):
                try:
                    response = self.session.post(
                        f"{self.base_url}/call_tool",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    result = response.json()
                    if not result["success"]:
                        raise Exception(f"Tool execution failed: {result.get('error', 'Unknown error')}")
                    return result["data"]
                except requests.RequestException as e:
                    if attempt == retries:
                        raise Exception(f"Failed to call tool after {retries + 1} attempts: {e}")
                    self.logger.warning(f"Attempt {attempt + 1} failed for {name}, retrying... ({e})")
                    time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            raise Exception(f"Failed to call tool: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "status": "unavailable"}
    
    def wait_for_server(self, max_wait=30):
        """Wait for server to become available"""
        self.logger.info("⏳ Waiting for server to start...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            health = self.health_check()
            if "error" not in health:
                self.logger.info("✅ Server is ready!")
                return True
            time.sleep(1)
        
        self.logger.error("❌ Server did not start within the timeout period")
        return False
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        try:
            response = self.session.get(f"{self.base_url}/tools")
            response.raise_for_status()
            return response.json()["tools"]
        except requests.RequestException as e:
            raise Exception(f"Failed to list tools: {e}")
    
    def call_tool(self, name: str, arguments: Dict[str, Any], retries=2) -> Dict[str, Any]:
        """Call a tool on the server with retry logic"""
        try:
            payload = {"name": name, "arguments": arguments}
            for attempt in range(retries + 1):
                try:
                    response = self.session.post(
                        f"{self.base_url}/call_tool",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    result = response.json()
                    if not result["success"]:
                        raise Exception(f"Tool execution failed: {result.get('error', 'Unknown error')}")
                    return result["data"]
                except requests.RequestException as e:
                    if attempt == retries:
                        raise Exception(f"Failed to call tool after {retries + 1} attempts: {e}")
                    self.logger.warning(f"Attempt {attempt + 1} failed for {name}, retrying... ({e})")
                    time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            raise Exception(f"Failed to call tool: {e}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get database schema information"""
        try:
            response = self.session.get(f"{self.base_url}/schema")
            response.raise_for_status()
            return response.json()["schema"]
        except requests.RequestException as e:
            raise Exception(f"Failed to get schema: {e}")
    
    def check_tool_status(self, tool_name: str) -> Dict[str, Any]:
        """Check the status of a tool execution (placeholder for future enhancement)"""
        try:
            response = self.session.get(f"{self.base_url}/tool_status/{tool_name}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Failed to check status for {tool_name}: {e}", "status": "unknown"}


class RateLimiter:
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    def can_proceed(self) -> bool:
        now = datetime.now()
        # Remove old requests
        while self.requests and (now - self.requests[0]) > timedelta(seconds=self.time_window):
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
    
    def wait_if_needed(self):
        while not self.can_proceed():
            time.sleep(1)