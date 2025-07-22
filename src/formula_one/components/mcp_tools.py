from abc import ABC, abstractmethod
from typing import Dict, Any
import time
from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.mcp_config_entity import ToolResult
from src.formula_one.entity.config_entity import DatabaseConfig

class BaseMCPTool(BaseComponent, ABC):
    """Base class for all MCP tools"""
    
    def __init__(self, config, db_config: DatabaseConfig, db_connection, query_builder):
        super().__init__(config, db_config)
        self.db_connection = db_connection
        self.query_builder = query_builder
    
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information for API documentation"""
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "",
            "category": getattr(self, 'category', 'general')
        }

class GetMeetingKeyTool(BaseMCPTool):
    """Get meeting key for a specific race event"""
    category = "core_identification"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            event_name = params.get("event_name")
            year = params.get("year")
            
            self.logger.info(f"Executing GetMeetingKeyTool for {event_name} {year}")
            
            query = self.query_builder.build_meeting_query(event_name, year)
            result = self.db_connection.execute_query(query)
            
            if result:
                row = result[0]
                data = {
                    "meeting_key": row[0],
                    "meeting_name": row[1],
                    "country_name": row[2],
                    "date": str(row[3]),
                    "year": row[4]
                }
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetMeetingKeyTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data=data,
                    sql_query=query,
                    sql_params={"event_name": event_name, "year": year},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetMeetingKeyTool: No meeting found for '{event_name}' in {year}")
                
                return ToolResult(
                    success=False,
                    error=f"No meeting found for '{event_name}' in {year}",
                    sql_query=query,
                    sql_params={"event_name": event_name, "year": year},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetMeetingKeyTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetSessionKeyTool(BaseMCPTool):
    """Get session key for a specific session type"""
    category = "core_identification"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            meeting_key = params.get("meeting_key")
            session_type = params.get("session_type")
            
            self.logger.info(f"Executing GetSessionKeyTool for meeting {meeting_key}, session {session_type}")
            
            query = self.query_builder.build_session_query(meeting_key, session_type)
            result = self.db_connection.execute_query(query)
            
            if result:
                row = result[0]
                data = {
                    "session_key": row[0],
                    "session_name": row[1],
                    "session_type": row[2],
                    "date_start": str(row[3]),
                    "date_end": str(row[4]) if row[4] else None,
                    "meeting_key": meeting_key
                }
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetSessionKeyTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data=data,
                    sql_query=query,
                    sql_params={"meeting_key": meeting_key, "session_type": session_type},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetSessionKeyTool: No {session_type} session found for meeting {meeting_key}")
                
                return ToolResult(
                    success=False,
                    error=f"No {session_type} session found for meeting {meeting_key}",
                    sql_query=query,
                    sql_params={"meeting_key": meeting_key, "session_type": session_type},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetSessionKeyTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetFastestLapTool(BaseMCPTool):
    """Get the fastest lap time and details for a specific session"""
    category = "performance_analysis"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_filter = params.get("driver_filter")
            team_filter = params.get("team_filter")
            
            self.logger.info(f"Executing GetFastestLapTool for session {session_key}")
            
            query = self.query_builder.build_fastest_lap_query(session_key, driver_filter, team_filter)
            result = self.db_connection.execute_query(query)
            
            if result:
                row = result[0]
                data = {
                    "driver": row[0],
                    "team": row[1],
                    "fastest_lap": float(row[2]) if row[2] else None,
                    "lap_number": row[3],
                    "sector1": float(row[4]) if row[4] else None,
                    "sector2": float(row[5]) if row[5] else None,
                    "sector3": float(row[6]) if row[6] else None,
                    "session_key": session_key
                }
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetFastestLapTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data=data,
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver_filter": driver_filter, "team_filter": team_filter},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetFastestLapTool: No fastest lap found for session {session_key}")
                
                return ToolResult(
                    success=False,
                    error="No fastest lap found",
                    sql_query=query,
                    sql_params={"session_key": session_key},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetFastestLapTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetDriverPerformanceTool(BaseMCPTool):
    """Get comprehensive performance data for a specific driver in a session"""
    category = "performance_analysis"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_name = params.get("driver_name")
            metrics = params.get("metrics", ["all"])
            
            self.logger.info(f"Executing GetDriverPerformanceTool for {driver_name} in session {session_key}")
            
            query = self.query_builder.build_driver_performance_query(session_key, driver_name, metrics)
            result = self.db_connection.execute_query(query)
            
            if result:
                performance_data = []
                for row in result:
                    performance_data.append({
                        "driver": row[0],
                        "team": row[1],
                        "total_laps": row[2],
                        "avg_lap": float(row[3]) if row[3] else None,
                        "best_lap": float(row[4]) if row[4] else None,
                        "worst_lap": float(row[5]) if row[5] else None,
                        "consistency": float(row[6]) if row[6] else None,
                        "position": row[7],
                        "incidents": row[8]
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetDriverPerformanceTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        "driver": driver_name,
                        "performance": performance_data,
                        "metrics": metrics
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver_name": driver_name, "metrics": metrics},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetDriverPerformanceTool: No performance data found for {driver_name}")
                
                return ToolResult(
                    success=False,
                    error=f"No performance data found for {driver_name}",
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver_name": driver_name},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetDriverPerformanceTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetTeamPerformanceTool(BaseMCPTool):
    """Get performance data for all drivers in a team for a session"""
    category = "performance_analysis"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            team_name = params.get("team_name")
            metrics = params.get("metrics", ["all"])
            
            self.logger.info(f"Executing GetTeamPerformanceTool for {team_name} in session {session_key}")
            
            query = self.query_builder.build_team_performance_query(session_key, team_name, metrics)
            result = self.db_connection.execute_query(query)
            
            if result:
                team_data = []
                for row in result:
                    team_data.append({
                        "driver": row[0],
                        "total_laps": row[1],
                        "avg_lap": float(row[2]) if row[2] else None,
                        "best_lap": float(row[3]) if row[3] else None,
                        "position": row[4],
                        "consistency": float(row[5]) if row[5] else None
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetTeamPerformanceTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        "team": team_name,
                        "drivers": team_data,
                        "metrics": metrics
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "team_name": team_name, "metrics": metrics},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetTeamPerformanceTool: No team data found for {team_name}")
                
                return ToolResult(
                    success=False,
                    error=f"No team data found for {team_name}",
                    sql_query=query,
                    sql_params={"session_key": session_key, "team_name": team_name},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetTeamPerformanceTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class CompareDriversTool(BaseMCPTool):
    """Compare performance between two drivers"""
    category = "comparison"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver1 = params.get("driver1")
            driver2 = params.get("driver2")
            
            self.logger.info(f"Executing CompareDriversTool for {driver1} vs {driver2} in session {session_key}")
            
            query = self.query_builder.build_driver_comparison_query(session_key, driver1, driver2)
            result = self.db_connection.execute_query(query)
            
            if result:
                driver_data = []
                for row in result:
                    driver_data.append({
                        "driver": row[0],
                        "team": row[1],
                        "total_laps": row[2],
                        "avg_lap": float(row[3]) if row[3] else None,
                        "best_lap": float(row[4]) if row[4] else None,
                        "worst_lap": float(row[5]) if row[5] else None,
                        "consistency": float(row[6]) if row[6] else None,
                        "incidents": row[7]
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"CompareDriversTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        "comparison": driver_data,
                        "driver1": driver1,
                        "driver2": driver2
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver1": driver1, "driver2": driver2},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"CompareDriversTool: No comparison data found for {driver1} vs {driver2}")
                
                return ToolResult(
                    success=False,
                    error=f"No comparison data found for {driver1} vs {driver2}",
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver1": driver1, "driver2": driver2},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"CompareDriversTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class CompareTeamsTool(BaseMCPTool):
    """Compare performance between two teams in a session"""
    category = "comparison"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            team1 = params.get("team1")
            team2 = params.get("team2")
            comparison_metrics = params.get("comparison_metrics", ["all"])
            
            self.logger.info(f"Executing CompareTeamsTool for {team1} vs {team2} in session {session_key}")
            
            query = self.query_builder.build_team_comparison_query(session_key, team1, team2, comparison_metrics)
            result = self.db_connection.execute_query(query)
            
            if result:
                team_data = []
                for row in result:
                    team_data.append({
                        "team": row[0],
                        "best_lap": float(row[1]) if row[1] else None,
                        "avg_lap": float(row[2]) if row[2] else None,
                        "consistency": float(row[3]) if row[3] else None,
                        "best_position": row[4],
                        "avg_position": float(row[5]) if row[5] else None
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"CompareTeamsTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        "comparison": team_data,
                        "team1": team1,
                        "team2": team2,
                        "metrics": comparison_metrics
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "team1": team1, "team2": team2},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"CompareTeamsTool: No comparison data found for {team1} vs {team2}")
                
                return ToolResult(
                    success=False,
                    error=f"No comparison data found for {team1} vs {team2}",
                    sql_query=query,
                    sql_params={"session_key": session_key, "team1": team1, "team2": team2},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"CompareTeamsTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetRaceResultsTool(BaseMCPTool):
    """Get race results with positions and details"""
    category = "results"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            result_type = params.get("result_type", "full_results")
            include_lap_times = params.get("include_lap_times", False)
            
            self.logger.info(f"Executing GetRaceResultsTool for session {session_key}, type {result_type}")
            
            query = self.query_builder.build_race_results_query(session_key, result_type, include_lap_times)
            result = self.db_connection.execute_query(query)
            
            if result:
                results = []
                for row in result:
                    result_dict = {
                        "driver": row[0],
                        "team": row[1],
                        "position": row[2]
                    }
                    if include_lap_times and len(row) > 3:
                        result_dict.update({
                            "best_lap": float(row[3]) if row[3] else None,
                            "avg_lap": float(row[4]) if row[4] else None,
                            "total_laps": row[5]
                        })
                    results.append(result_dict)
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetRaceResultsTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        "result_type": result_type,
                        "results": results
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "result_type": result_type},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetRaceResultsTool: No race results found for session {session_key}")
                
                return ToolResult(
                    success=False,
                    error="No race results found",
                    sql_query=query,
                    sql_params={"session_key": session_key},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetRaceResultsTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetQualifyingResultsTool(BaseMCPTool):
    """Get qualifying results with best lap times and positions"""
    category = "results"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            result_type = params.get("result_type", "full_results")
            
            self.logger.info(f"Executing GetQualifyingResultsTool for session {session_key}, type {result_type}")
            
            query = self.query_builder.build_qualifying_results_query(session_key, result_type)
            result = self.db_connection.execute_query(query)
            
            if result:
                results = []
                for row in result:
                    results.append({
                        "driver": row[0],
                        "team": row[1],
                        "position": row[2],
                        "best_lap": float(row[3]) if row[3] else None,
                        "q1_time": float(row[4]) if row[4] else None,
                        "q2_time": float(row[5]) if row[5] else None,
                        "q3_time": float(row[6]) if row[6] else None
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetQualifyingResultsTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        "result_type": result_type,
                        "results": results
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "result_type": result_type},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetQualifyingResultsTool: No qualifying results found for session {session_key}")
                
                return ToolResult(
                    success=False,
                    error="No qualifying results found",
                    sql_query=query,
                    sql_params={"session_key": session_key},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetQualifyingResultsTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetPitStopAnalysisTool(BaseMCPTool):
    """Get pit stop strategy and performance for drivers/teams"""
    category = "strategy"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_filter = params.get("driver_filter")
            team_filter = params.get("team_filter")
            analysis_type = params.get("analysis_type", "all")
            
            self.logger.info(f"Executing GetPitStopAnalysisTool for session {session_key}")
            
            query = self.query_builder.build_pit_stop_analysis_query(session_key, driver_filter, team_filter, analysis_type)
            result = self.db_connection.execute_query(query)
            
            if result:
                pit_stops = []
                for row in result:
                    pit_stops.append({
                        "driver": row[0],
                        "team": row[1],
                        "lap": row[2],
                        "duration": float(row[3]) if row[3] else None,
                        "stop_number": row[4],
                        "tire_compound": row[5] if len(row) > 5 else None
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetPitStopAnalysisTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        analysis_type: analysis_type,
                        "pit_stops": pit_stops
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver_filter": driver_filter, "team_filter": team_filter},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetPitStopAnalysisTool: No pit stop data found for session {session_key}")
                
                return ToolResult(
                    success=False,
                    error="No pit stop data found",
                    sql_query=query,
                    sql_params={"session_key": session_key},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetPitStopAnalysisTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetTireStrategyTool(BaseMCPTool):
    """Analyze tire strategy and stint information for drivers/teams"""
    category = "strategy"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_filter = params.get("driver_filter")
            team_filter = params.get("team_filter")
            strategy_type = params.get("strategy_type", "all")
            
            self.logger.info(f"Executing GetTireStrategyTool for session {session_key}")
            
            query = self.query_builder.build_tire_strategy_query(session_key, driver_filter, team_filter, strategy_type)
            result = self.db_connection.execute_query(query)
            
            if result:
                stints = []
                for row in result:
                    stints.append({
                        "driver": row[0],
                        "team": row[1],
                        "stint_number": row[2],
                        "start_lap": row[3],
                        "end_lap": row[4],
                        "tire_compound": row[5],
                        "stint_length": row[6],
                        "avg_lap_time": float(row[7]) if row[7] else None
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetTireStrategyTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        strategy_type: strategy_type,
                        "stints": stints
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver_filter": driver_filter, "team_filter": team_filter},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetTireStrategyTool: No tire strategy data found for session {session_key}")
                
                return ToolResult(
                    success=False,
                    error="No tire strategy data found",
                    sql_query=query,
                    sql_params={"session_key": session_key},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetTireStrategyTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class InvestigateIncidentTool(BaseMCPTool):
    """Investigate incidents, slow laps, or unusual performance patterns"""
    category = "incident_analysis"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_name = params.get("driver_name")
            lap_number = params.get("lap_number")
            investigation_type = params.get("investigation_type", "all")
            context_laps = params.get("context_laps", 3)
            
            self.logger.info(f"Executing InvestigateIncidentTool for {driver_name} lap {lap_number}")
            
            query = self.query_builder.build_incident_investigation_query(session_key, driver_name, lap_number, investigation_type, context_laps)
            result = self.db_connection.execute_query(query)
            
            if result:
                incident_data = []
                for row in result:
                    incident_data.append({
                        "lap": row[0],
                        "lap_time": float(row[1]) if row[1] else None,
                        "sector1": float(row[2]) if row[2] else None,
                        "sector2": float(row[3]) if row[3] else None,
                        "sector3": float(row[4]) if row[4] else None,
                        "position": row[5],
                        "incident": row[6] if len(row) > 6 else None
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"InvestigateIncidentTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        "driver": driver_name,
                        "target_lap": lap_number,
                        "investigation_type": investigation_type,
                        "context_laps": context_laps,
                        "lap_data": incident_data
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver_name": driver_name, "lap_number": lap_number},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"InvestigateIncidentTool: No incident data found for {driver_name} lap {lap_number}")
                
                return ToolResult(
                    success=False,
                    error=f"No incident data found for {driver_name} lap {lap_number}",
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver_name": driver_name, "lap_number": lap_number},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"InvestigateIncidentTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetPositionProgressionTool(BaseMCPTool):
    """Get position changes throughout the session for drivers"""
    category = "position_analysis"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_filter = params.get("driver_filter")
            team_filter = params.get("team_filter")
            progression_type = params.get("progression_type", "all")
            
            self.logger.info(f"Executing GetPositionProgressionTool for session {session_key}")
            
            query = self.query_builder.build_position_progression_query(session_key, driver_filter, team_filter, progression_type)
            result = self.db_connection.execute_query(query)
            
            if result:
                progression_data = []
                for row in result:
                    progression_data.append({
                        "driver": row[0],
                        "team": row[1],
                        "lap": row[2],
                        "position": row[3],
                        "position_change": row[4] if len(row) > 4 else None
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetPositionProgressionTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        "progression_type": progression_type,
                        "progression": progression_data
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver_filter": driver_filter, "team_filter": team_filter},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetPositionProgressionTool: No position progression data found for session {session_key}")
                
                return ToolResult(
                    success=False,
                    error="No position progression data found",
                    sql_query=query,
                    sql_params={"session_key": session_key},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetPositionProgressionTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetSectorAnalysisTool(BaseMCPTool):
    """Analyze sector times and identify strengths/weaknesses"""
    category = "sector_analysis"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            driver_filter = params.get("driver_filter")
            team_filter = params.get("team_filter")
            sector_analysis_type = params.get("sector_analysis_type", "all")
            
            self.logger.info(f"Executing GetSectorAnalysisTool for session {session_key}")
            
            query = self.query_builder.build_sector_analysis_query(session_key, driver_filter, team_filter, sector_analysis_type)
            result = self.db_connection.execute_query(query)
            
            if result:
                sector_data = []
                for row in result:
                    sector_data.append({
                        "driver": row[0],
                        "team": row[1],
                        "best_sector1": float(row[2]) if row[2] else None,
                        "best_sector2": float(row[3]) if row[3] else None,
                        "best_sector3": float(row[4]) if row[4] else None,
                        "avg_sector1": float(row[5]) if row[5] else None,
                        "avg_sector2": float(row[6]) if row[6] else None,
                        "avg_sector3": float(row[7]) if row[7] else None
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetSectorAnalysisTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        "sector_analysis_type": sector_analysis_type,
                        "sector_data": sector_data
                    },
                    sql_query=query,
                    sql_params={"session_key": session_key, "driver_filter": driver_filter, "team_filter": team_filter},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetSectorAnalysisTool: No sector analysis data found for session {session_key}")
                
                return ToolResult(
                    success=False,
                    error="No sector analysis data found",
                    sql_query=query,
                    sql_params={"session_key": session_key},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetSectorAnalysisTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class ExploreSchemaTool(BaseMCPTool):
    """Get information about database tables, columns, and relationships"""
    category = "utility"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            table_name = params.get("table_name")
            detail_level = params.get("detail_level", "overview")
            
            self.logger.info(f"Executing ExploreSchemaTool for table {table_name}, detail {detail_level}")
            
            query = self.query_builder.build_schema_exploration_query(table_name, detail_level)
            result = self.db_connection.execute_query(query)
            
            if result:
                schema_data = []
                for row in result:
                    schema_data.append({
                        "table_name": row[0] if len(row) > 0 else None,
                        "column_name": row[1] if len(row) > 1 else None,
                        "data_type": row[2] if len(row) > 2 else None,
                        "is_nullable": row[3] if len(row) > 3 else None,
                        "sample_value": row[4] if len(row) > 4 else None
                    })
                
                execution_time = time.time() - start_time
                self.logger.info(f"ExploreSchemaTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data={
                        "table_name": table_name,
                        "detail_level": detail_level,
                        "schema_info": schema_data
                    },
                    sql_query=query,
                    sql_params={"table_name": table_name, "detail_level": detail_level},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"ExploreSchemaTool: No schema data found for table {table_name}")
                
                return ToolResult(
                    success=False,
                    error=f"No schema data found for table {table_name}",
                    sql_query=query,
                    sql_params={"table_name": table_name},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"ExploreSchemaTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class GetSessionInfoTool(BaseMCPTool):
    """Get basic information about a session including type, date, and participants"""
    category = "utility"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            session_key = params.get("session_key")
            
            self.logger.info(f"Executing GetSessionInfoTool for session {session_key}")
            
            query = self.query_builder.build_session_info_query(session_key)
            result = self.db_connection.execute_query(query)
            
            if result:
                row = result[0]
                data = {
                    "session_key": session_key,
                    "session_name": row[0],
                    "session_type": row[1],
                    "meeting_name": row[2],
                    "country": row[3],
                    "date": str(row[4]),
                    "total_drivers": row[5],
                    "total_laps": row[6] if len(row) > 6 else None
                }
                
                execution_time = time.time() - start_time
                self.logger.info(f"GetSessionInfoTool completed successfully in {execution_time:.3f}s")
                
                return ToolResult(
                    success=True,
                    data=data,
                    sql_query=query,
                    sql_params={"session_key": session_key},
                    execution_time=execution_time
                )
            else:
                execution_time = time.time() - start_time
                self.logger.warning(f"GetSessionInfoTool: No session info found for session {session_key}")
                
                return ToolResult(
                    success=False,
                    error=f"No session info found for session {session_key}",
                    sql_query=query,
                    sql_params={"session_key": session_key},
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"GetSessionInfoTool failed: {e}")
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

# # Tool registry for easy access
# MCP_TOOLS = {
#     "get_meeting_key": GetMeetingKeyTool,
#     "get_session_key": GetSessionKeyTool,
#     "get_fastest_lap": GetFastestLapTool,
#     "get_driver_performance": GetDriverPerformanceTool,
#     "get_team_performance": GetTeamPerformanceTool,
#     "compare_drivers": CompareDriversTool,
#     "compare_teams": CompareTeamsTool,
#     "get_race_results": GetRaceResultsTool,
#     "get_qualifying_results": GetQualifyingResultsTool,
#     "get_pit_stop_analysis": GetPitStopAnalysisTool,
#     "get_tire_strategy": GetTireStrategyTool,
#     "investigate_incident": InvestigateIncidentTool,
#     "get_position_progression": GetPositionProgressionTool,
#     "get_sector_analysis": GetSectorAnalysisTool,
#     "explore_schema": ExploreSchemaTool,
#     "get_session_info": GetSessionInfoTool
# }