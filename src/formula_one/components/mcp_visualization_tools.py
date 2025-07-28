from abc import ABC, abstractmethod
from typing import Dict, Any
import time
from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.mcp_config_entity import ToolResult
from src.formula_one.entity.config_entity import DatabaseConfig
from src.formula_one.components.generate_visualizations import VisualizationGenerator

class BaseVisualizationTool(BaseComponent, ABC):
    """Base class for visualization tools"""
    
    def __init__(self, config, db_config: DatabaseConfig, db_connection, query_builder):
        super().__init__(config, db_config)
        self.db_connection = db_connection
        self.query_builder = query_builder
        self.viz_generator = VisualizationGenerator(config, db_config)
        # Set the query builder for the visualization generator
        self.viz_generator.query_builder = query_builder
    
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the visualization tool with given parameters"""
        pass
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information for API documentation"""
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "",
            "category": "visualization"
        }

class CreateLapTimeProgressionTool(BaseVisualizationTool):
    """Create lap time progression visualization for a session"""
    category = "visualization"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_filter = params.get("driver_filter")
            team_filter = params.get("team_filter")
            
            self.logger.info(f"Executing CreateLapTimeProgressionTool for session {session_key}")
            
            result = self.viz_generator.create_lap_time_progression(
                session_key, driver_filter, team_filter
            )
            
            execution_time = time.time() - start_time
            self.logger.info(f"CreateLapTimeProgressionTool completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"CreateLapTimeProgressionTool failed: {e}")
            
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                execution_time=execution_time
            )

class CreatePositionProgressionTool(BaseVisualizationTool):
    """Create position progression visualization for a session"""
    category = "visualization"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_filter = params.get("driver_filter")
            team_filter = params.get("team_filter")
            
            self.logger.info(f"Executing CreatePositionProgressionTool for session {session_key} with driver_filter: {driver_filter}, team_filter: {team_filter}")
            
            result = self.viz_generator.create_position_progression(session_key, driver_filter, team_filter)
            
            execution_time = time.time() - start_time
            self.logger.info(f"CreatePositionProgressionTool completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"CreatePositionProgressionTool failed: {e}")
            
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                execution_time=execution_time
            )

class CreateSectorAnalysisTool(BaseVisualizationTool):
    """Create sector analysis visualization for a session"""
    category = "visualization"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_filter = params.get("driver_filter")
            
            self.logger.info(f"Executing CreateSectorAnalysisTool for session {session_key}")
            
            result = self.viz_generator.create_sector_analysis(session_key, driver_filter)
            
            execution_time = time.time() - start_time
            self.logger.info(f"CreateSectorAnalysisTool completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"CreateSectorAnalysisTool failed: {e}")
            
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                execution_time=execution_time
            )

class CreatePitStopAnalysisTool(BaseVisualizationTool):
    """Create pit stop analysis visualization for a session"""
    category = "visualization"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_filter = params.get("driver_filter")
            team_filter = params.get("team_filter")
            analysis_type = params.get("analysis_type", "comprehensive")  # Default to comprehensive
            
            self.logger.info(f"Executing CreatePitStopAnalysisTool for session {session_key} with analysis_type: {analysis_type}")
            
            result = self.viz_generator.create_pit_stop_analysis(
                session_key, driver_filter, team_filter, analysis_type
            )
            
            execution_time = time.time() - start_time
            self.logger.info(f"CreatePitStopAnalysisTool completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"CreatePitStopAnalysisTool failed: {e}")
            
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                execution_time=execution_time
            )

class CreateTireStrategyTool(BaseVisualizationTool):
    """Create tire strategy visualization for a session"""
    category = "visualization"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_filter = params.get("driver_filter")
            team_filter = params.get("team_filter")
            
            self.logger.info(f"Executing CreateTireStrategyTool for session {session_key}")
            
            result = self.viz_generator.create_tire_strategy_visualization(
                session_key, driver_filter, team_filter
            )
            
            execution_time = time.time() - start_time
            self.logger.info(f"CreateTireStrategyTool completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"CreateTireStrategyTool failed: {e}")
            
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                execution_time=execution_time
            )