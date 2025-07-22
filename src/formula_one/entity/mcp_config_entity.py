from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MCPConfig:
    """Configuration for MCP server and client"""
    server_host: str = "localhost"
    server_port: int = 8000
    openai_api_key: str = None
    langsmith_api_key: str = None
    langsmith_project: str = "f1-mcp-bot"
    
    def __post_init__(self):
        self.openai_api_key = self.openai_api_key or os.getenv('OPENAI_API_KEY')
        self.langsmith_api_key = self.langsmith_api_key or os.getenv('LANGSMITH_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

@dataclass
class ToolConfig:
    """Configuration for MCP tools"""
    enable_sql_logging: bool = True
    enable_performance_metrics: bool = True
    max_query_results: int = 100
    default_session_type: str = "Race"

@dataclass
class ToolResult:
    """Standardized tool result"""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    sql_query: Optional[str] = None
    sql_params: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

@dataclass
class MCPTool:
    """Represents an MCP tool with its capabilities"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    examples: List[Dict[str, Any]]
    category: str
    implemented: bool = True