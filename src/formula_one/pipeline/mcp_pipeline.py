from src.formula_one.logging import logger
from src.formula_one.entity.mcp_config_entity import MCPConfig
from src.formula_one.entity.config_entity import DatabaseConfig
from src.formula_one.components.mcp_tools import (
    GetMeetingKeyTool, GetSessionKeyTool, GetDriverPerformanceTool, GetTeamPerformanceTool, CompareDriversTool,CompareTeamsTool, GetRaceResultsTool,
    GetQualifyingResultsTool, GetPitStopAnalysisTool, GetTireStrategyTool, InvestigateIncidentTool, GetPositionProgressionTool, GetSectorAnalysisTool,
    GetSessionInfoTool, ExploreSchemaTool, GetFastestLapTool
)
from src.formula_one.components.mcp_server import FastAPIServer
from src.formula_one.components.mcp_client import HTTPMCPClient
from src.formula_one.components.mcp_reasoning import ReasoningEngine
from src.formula_one.utils.database_utils import DatabaseUtils
from src.formula_one.config.configuration import ConfigurationManager

import time
import signal
import sys
import os
from dotenv import load_dotenv

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("_client").setLevel(logging.WARNING)

class MCPTrainingPipeline:
    """Main training pipeline for MCP system"""
    
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_mcp_config()
        self.db_config = DatabaseConfig()
        self.logger = logger
        self.query_builder = config_manager.get_query_builder()
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize database components
        self.db_utils = DatabaseUtils(self.db_config)
        self.schema_info = self.db_utils.get_schema_info()
        
        # Initialize tools
        self.tools = {
            "get_meeting_key": GetMeetingKeyTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_session_key": GetSessionKeyTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_fastest_lap": GetFastestLapTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_driver_performance": GetDriverPerformanceTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_team_performance": GetTeamPerformanceTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "compare_drivers": CompareDriversTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "compare_teams": CompareTeamsTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_race_results": GetRaceResultsTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_qualifying_results": GetQualifyingResultsTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_pit_stop_analysis": GetPitStopAnalysisTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_tire_strategy": GetTireStrategyTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "investigate_incident": InvestigateIncidentTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_position_progression": GetPositionProgressionTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_sector_analysis": GetSectorAnalysisTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "explore_schema": ExploreSchemaTool(self.config, self.db_config, self.db_utils, self.query_builder),
            "get_session_info": GetSessionInfoTool(self.config, self.db_config, self.db_utils, self.query_builder)
        }
        
        # Initialize server
        self.server = FastAPIServer(self.config, self.db_config, self.tools)
        
        # Initialize client
        self.client = HTTPMCPClient(self.config, self.db_config)
        
        # Initialize reasoning engine
        self.reasoning_engine = ReasoningEngine(self.config, self.db_config, self.tools, self.client)
        
        self.logger.info("MCP Training Pipeline initialized successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Received shutdown signal, cleaning up...")
        self.running = False
        if hasattr(self, 'server'):
            self.server.stop_server()
        sys.exit(0)
    
    def initiate_mcp_system(self):
        """Initialize the complete MCP system and start interactive testing"""
        self.logger.info("Starting MCP training pipeline")
        
        try:
            # Test database connection
            self.logger.info("Testing database connection...")
            test_result = self.db_utils.execute_mcp_query("SELECT 1")
            self.logger.info("Database connection successful")
            
            # Start the server
            self.start_server()
            
            # Wait for server to be ready
            if not self.client.wait_for_server(max_wait=30):
                raise Exception("Server failed to start")
            
            self.logger.info("MCP system initialized successfully. Starting interactive test mode...")
            print(f"F1 Bot is ready! Enter a query (or 'exit' to quit) at {time.strftime('%H:%M:%S %Z on %Y-%m-%d', time.localtime())}:")
            # while True:
            #     query = input("> ").strip()
            #     if query.lower() == "exit":
            #         break
            #     if query:
            #         response = self.test_reasoning_engine(query)
            #         print(f"Response: {response}\n")
            
            while self.running:
                query = input("> ").strip()
                if query.lower() == "exit":
                    break
                if query:
                    response = self.test_reasoning_engine(query)
                    print(f"Response: {response}\n")       

        except Exception as e:
            self.logger.info("Received keyboard interrupt")
        finally:
            self.server.stop_server()
    
    def start_server(self):
        """Start the MCP server"""
        self.logger.info("Starting MCP server...")
        self.server.run_server_in_thread()
    
    def test_reasoning_engine(self, query: str):
        """Test the reasoning engine with a query"""
        #self.logger.info(f"Testing reasoning engine with query: {query}")
        response = self.reasoning_engine.reason_and_answer(query)
        #self.logger.info(f"Reasoning engine response: {response}")
        return response