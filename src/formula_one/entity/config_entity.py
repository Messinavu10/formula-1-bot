from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
load_dotenv()

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion from OpenF1 API"""
    source_url: str = None
    api_timeout: int = None
    max_retries: int = None
    batch_size: int = None
    target_meetings: List[Dict[str, str]] = None
    sessions_to_ingest: List[str] = None
    endpoints: List[str] = None
    
    def __post_init__(self):
        # Convert to proper types (from config.yaml)
        if self.api_timeout:
            self.api_timeout = int(self.api_timeout)
        if self.max_retries:
            self.max_retries = int(self.max_retries)
        if self.batch_size:
            self.batch_size = int(self.batch_size)
        
        if self.target_meetings is None:
            self.target_meetings = [{"meeting_key": "latest"}]
        if self.sessions_to_ingest is None:
            self.sessions_to_ingest = [
                "Practice 1", "Practice 2", "Practice 3", 
                "Qualifying", "Sprint", "Race"
            ]
        if self.endpoints is None:
            self.endpoints = [
                "meetings", "sessions", "drivers", "laps", 
                "pit", "stints", "position", 
                "intervals", "race_control", 
                "weather"
            ]

@dataclass
class DatabaseConfig:
    """Configuration for database using environment variables"""
    host: str = None
    port: int = None
    database: str = None
    user: str = None
    password: str = None
    
    def __post_init__(self):
        # Load from environment variables only - no hardcoded defaults
        self.host = self.host or os.getenv('DB_HOST')
        self.port = int(self.port or os.getenv('DB_PORT')) if (self.port or os.getenv('DB_PORT')) else None
        self.database = self.database or os.getenv('DB_NAME')
        self.user = self.user or os.getenv('DB_USER')
        self.password = self.password or os.getenv('DB_PASSWORD')
        
        # Validate database config
        self.validate_config()
    
    def validate_config(self):
        """Validate that all required database configuration is present"""
        if not self.host:
            raise ValueError("DB_HOST environment variable is required")
        if not self.port:
            raise ValueError("DB_PORT environment variable is required")
        if not self.database:
            raise ValueError("DB_NAME environment variable is required")
        if not self.user:
            raise ValueError("DB_USER environment variable is required")
        if not self.password:
            raise ValueError("DB_PASSWORD environment variable is required")
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"