import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import time
import os
import numpy as np

from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.config_entity import DatabaseConfig
from src.formula_one.entity.mcp_config_entity import MCPConfig
from sentence_transformers import SentenceTransformer


class IntentAnalyzer(BaseComponent):
    """Analyze user query intent and extract information"""
    
    def __init__(self, config, db_config: DatabaseConfig):
        super().__init__(config, db_config)

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define intent patterns with examples
        self.intent_patterns = {
            "comparison": [
                "compare", "vs", "versus", "difference", "who was better",
                "how did X compare to Y", "which driver was faster", "compare X and Y"
            ],
            "driver_performance": [
                "driver performance", "how did X perform", "what was X's performance",
                "driver analysis", "driver stats", "driver data"
            ],
            "team_performance": [
                "team performance", "how did X team perform", "team analysis",
                "team stats", "team data", "mercedes team performance", "ferrari team performance",
                "mclaren team performance", "red bull team performance", "williams team performance"
            ],
            "fastest_lap": [
                "fastest lap", "best lap", "quickest", "fastest time",
                "best lap time", "quickest lap"
            ],
            "sector_analysis": [
                "sector", "sectors", "sector times", "sector analysis", "sector performance",
                "sector 1", "sector 2", "sector 3", "first sector", "second sector", "third sector",
                "sector strengths", "sector weaknesses", "best sector", "worst sector",
                "sector consistency", "sector comparison", "sector breakdown"
            ],
            "tire_strategy": [
                "tire strategy", "tyre strategy", "compound", "tire management",
                "tyre management", "stints", "tire compounds", "tyre compounds"
            ],
            "pit_strategy": [
                "pit stop", "pitstop", "stops", "fastest pitstop",
                "pit strategy", "when did they pit", "how many stops"
            ],
            "incident_investigation": [
                "incident", "accident", "crash", "what happened", "what happened on lap",
                "incident on lap", "crash on lap", "accident on lap", "what went wrong",
                "problem on lap", "issue on lap", "slow lap", "unusual", "strange"
            ],
            "qualifying_results": [
                "qualifying", "quali", "grid", "pole", "qualifying results",
                "starting position", "grid position"
            ],
            "race_results": [
                "race results", "who won", "podium", "finishing position",
                "race outcome", "final results"
            ],
            "visualization": [
                "show me a graph", "create a chart", "visualize", "plot", "graph",
                "show me the data", "display", "chart", "visualization", "graphical",
                "show me a visualization", "create a plot", "show the trend",
                "plot the data", "graph the results", "show me the progression",
                "visualize the data", "create a diagram", "show me the analysis",
                "show me"
            ]
        }
        
        # Pre-compute embeddings for all patterns
        self.pattern_embeddings = {}
        for intent, patterns in self.intent_patterns.items():
            self.pattern_embeddings[intent] = self.model.encode(patterns)
    
    def analyze_query_intent(self, user_query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Analyze what the user is asking for with conversation context"""
        query_lower = user_query.lower()
        
        # Extract meeting info
        meeting_info = self._extract_meeting_info(user_query)
        
        # If no meeting found, try to get from conversation history
        if not meeting_info["name"] and conversation_history:
            meeting_info = self._get_context_from_history(conversation_history)
        
        # Determine session type
        session_type = self._determine_session_type(query_lower)
        
        # Determine query type
        query_type = self._determine_query_type(query_lower)
        

        # Extract lap number for incident investigation
        lap_number = self._extract_lap_number(user_query) if query_type == "incident_investigation" else None

        return {
            "meeting_info": meeting_info,
            "session_type": session_type,
            "query_type": query_type,
            "drivers": self._extract_driver_names(user_query),
            "lap_number": lap_number
        }
    
    def _extract_lap_number(self, query: str) -> Optional[int]:
        """Extract lap number from query"""
        import re
        query_lower = query.lower()
        
        # Look for patterns like "lap 10", "lap 5", etc.
        lap_patterns = [
            r'lap\s+(\d+)',
            r'on\s+lap\s+(\d+)',
            r'incident\s+on\s+lap\s+(\d+)',
            r'what\s+happened\s+on\s+lap\s+(\d+)',
            r'crash\s+on\s+lap\s+(\d+)',
            r'accident\s+on\s+lap\s+(\d+)',
            r'problem\s+on\s+lap\s+(\d+)',
            r'issue\s+on\s+lap\s+(\d+)'
        ]
        
        for pattern in lap_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_meeting_info(self, query: str) -> Dict[str, Any]:
        """Extract meeting name and year from query"""
        race_keywords = {
            'australian': 'Australian Grand Prix',
            'chinese': 'Chinese Grand Prix',
            'china': 'Chinese Grand Prix',
            'japanese': 'Japanese Grand Prix',
            'japan': 'Japanese Grand Prix',
            'bahrain': 'Bahrain Grand Prix',
            'saudi': 'Saudi Arabian Grand Prix',
            'saudi arabian': 'Saudi Arabian Grand Prix',
            'miami': 'Miami Grand Prix',
            'italian emilia-romagna': 'Emilia‑Romagna Grand Prix',
            'emilia romagna': 'Emilia‑Romagna Grand Prix',
            'emilia-romagna': 'Emilia‑Romagna Grand Prix',
            'imola': 'Emilia‑Romagna Grand Prix',
            'monaco': 'Monaco Grand Prix',
            'spanish': 'Spanish Grand Prix',
            'spain': 'Spanish Grand Prix',
            'canadian': 'Canadian Grand Prix',
            'canada': 'Canadian Grand Prix', 
            'austrian': 'Austrian Grand Prix',
            'austria': 'Austrian Grand Prix',
            'british': 'British Grand Prix',
            'silverstone': 'British Grand Prix',
            'great britain': 'British Grand Prix',
            'belgian' : 'Belgian Grand Prix'
        }
        
        query_lower = query.lower()
        meeting_name = None
        
        for keyword, race_name in race_keywords.items():
            if keyword in query_lower:
                meeting_name = race_name
                break
        
        # Extract year (default to 2025)
        year_match = re.search(r'20\d{2}', query)
        year = int(year_match.group()) if year_match else 2025
        
        return {
            "name": meeting_name,
            "year": year
        }
    
    def _get_context_from_history(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Get context from recent conversation history"""
        for entry in reversed(conversation_history[-3:]):
            previous_meeting = self._extract_meeting_info(entry["message"])
            if previous_meeting["name"]:
                self.logger.info(f"Using context from previous query: {previous_meeting['name']}")
                return previous_meeting
        return {"name": None, "year": 2025}
    
    def _determine_session_type(self, query_lower: str) -> str:
        """Determine session type from query"""
        if any(word in query_lower for word in ["qualifying", "quali", "q1", "q2", "q3", "pole", "grid"]):
            return "Qualifying"
        elif any(word in query_lower for word in ["sprint race", "sprint"]) and "qualifying" not in query_lower:
            return "Sprint"
        elif any(word in query_lower for word in ["practice", "fp1", "fp2", "fp3", "free practice"]):
            if "1" in query_lower or "fp1" in query_lower:
                return "Practice 1"
            elif "2" in query_lower or "fp2" in query_lower:
                return "Practice 2"
            elif "3" in query_lower or "fp3" in query_lower:
                return "Practice 3"
            else:
                return "Practice"
        else:
            return "Race"

    def _determine_query_type(self, query_lower: str) -> str:
        """Determine query type using semantic similarity with priority logic"""
        
        # First, check for explicit comparison keywords (highest priority)
        comparison_keywords = ["compare", "vs", "versus", "difference", "who was better", "which driver was faster"]
        for keyword in comparison_keywords:
            if keyword in query_lower:
                return "comparison"
        
        # Second, check for visualization keywords (high priority)
        viz_keywords = [
            "show me a graph", "create a chart", "visualize", "plot", "graph",
            "show me the data", "display", "chart", "visualization", "graphical",
            "show me a visualization", "create a plot", "show the trend",
            "plot the data", "graph the results", "show me the progression",
            "visualize the data", "create a diagram", "show me the analysis",
            "show me"
        ]
        
        for keyword in viz_keywords:
            if keyword in query_lower:
                return "visualization"
        
        # Third, check for pit stop keywords (lower priority)
        pit_stop_keywords = [
            "pit stop", "pitstop", "pitted", "when did they pit", "pit strategy",
            "pit stops", "pitstop", "when did", "pit timing", "pit duration",
            "pit stop analysis", "pit stop strategy", "pit stop timing",
            "when drivers pitted", "pit stop times", "pit stop performance"
        ]
        for keyword in pit_stop_keywords:
            if keyword in query_lower:
                return "pit_strategy"
        
        # Check for lap-related incident keywords (more flexible)
        lap_incident_keywords = [
            "what happened on lap",
            "what happened to",
            "incident on lap", 
            "crash on lap", 
            "accident on lap",
            "what happened",
            "incident",
            "crash",
            "accident"
        ]
        
        # Check if query contains both lap number and incident keywords
        import re
        has_lap_number = bool(re.search(r'\blap\s+\d+', query_lower))
        has_incident_keyword = any(keyword in query_lower for keyword in lap_incident_keywords)
        
        if has_lap_number and has_incident_keyword:
            return "incident_investigation"
        
        # Check for sector analysis keywords
        sector_keywords = [
            "sector", "sectors", "sector times", "sector analysis", "sector performance",
            "sector 1", "sector 2", "sector 3", "first sector", "second sector", "third sector",
            "sector strengths", "sector weaknesses", "best sector", "worst sector",
            "sector consistency", "sector comparison", "sector breakdown"
        ]
        for keyword in sector_keywords:
            if keyword in query_lower:
                return "sector_analysis"
        
        # Check for explicit team performance keywords
        team_keywords = ["team performance", "team analysis", "team stats", "team data"]
        for keyword in team_keywords:
            if keyword in query_lower:
                return "team_performance"
        
        # Check for driver performance keywords
        driver_keywords = ["driver performance", "driver analysis", "driver stats", "driver data"]
        for keyword in driver_keywords:
            if keyword in query_lower:
                return "driver_performance"
        
        # If no explicit keywords, use semantic similarity
        query_embedding = self.model.encode([query_lower])
        
        best_intent = "race_results"  # default
        best_score = 0
        
        # Compare query with each intent pattern
        for intent, pattern_embeddings in self.pattern_embeddings.items():
            # Calculate cosine similarity between query and all patterns for this intent
            similarities = np.dot(pattern_embeddings, query_embedding.T).flatten()
            max_similarity = np.max(similarities)
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_intent = intent
        
        # Set a threshold - if no good match, default to race_results
        if best_score < 0.3:  # Adjust threshold as needed
            return "race_results"

        return best_intent
    
    def _extract_driver_names(self, query: str) -> List[str]:
        """Extract driver names from query"""
        driver_mappings = {
            "Oscar PIASTRI": ["oscar", "piastri", "oscar piastri"],
            "Lando NORRIS": ["lando", "norris", "lando norris"],
            "George RUSSELL": ["george", "russell", "george russell"],
            "Andrea Kimi ANTONELLI": ["antonelli", "kimi", "kimi antonelli", "andrea antonelli"],
            "Max VERSTAPPEN": ["max", "verstappen", "max verstappen"],
            "Liam LAWSON": ["liam", "lawson", "liam lawson"],
            "Charles LECLERC": ["charles", "leclerc", "charles leclerc"],
            "Lewis HAMILTON": ["lewis", "hamilton", "lewis hamilton"],
            "Esteban OCON": ["esteban", "ocon", "esteban ocon"],
            "Oliver BEARMAN": ["oliver", "bearman", "oliver bearman"],
            "Alex ALBON": ["alex", "albon", "alex albon"],
            "Carlos SAINZ": ["carlos", "sainz", "carlos sainz"],
            "Pierre GASLY": ["pierre", "gasly", "pierre gasly"],
            "Jack DOOHAN": ["jack", "doohan", "jack doohan"],
            "Lance STROLL": ["lance", "stroll", "lance stroll"],
            "Fernando ALONSO": ["fernando", "alonso", "fernando alonso"],
            "Isack HADJAR": ["isack", "hadjar", "isack hadjar"],
            "Yuki TSUNODA": ["yuki", "tsunoda", "yuki tsunoda"],
            "Gabriel BORTOLETO": ["gabriel", "bortoleto", "gabriel bortoleto"],
            "Nico HULKENBERG": ["nico", "hulkenberg", "nico hulkenberg", "hulk"]
        }
        
        query_lower = query.lower()
        found_drivers = []
        
        for full_name, aliases in driver_mappings.items():
            for alias in aliases:
                if alias in query_lower:
                    if full_name not in found_drivers:
                        found_drivers.append(full_name)
                    break
        
        return found_drivers 

class ContextManager(BaseComponent):
    """Manage conversation context and clarification handling"""
    
    def __init__(self, config, db_config: DatabaseConfig):
        super().__init__(config, db_config)
        self.conversation_history = []
        self.last_clarification_context = None
        self.last_meeting_context = None  # ← Add this like your notebook
    
    def add_to_history(self, message: str, response: str):
        """Add to conversation history"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response
        })
    
    def is_clarification_response(self, user_query: str) -> bool:
        """Check if this is a clarification response"""
        if not self.conversation_history:
            return False
        
        last_response = self.conversation_history[-1]["response"]
        # ← Use the same logic as your notebook
        return "Which one would you like to know about?" in last_response and "Sprint" in last_response
    
    def get_clarification_context(self) -> Optional[Dict[str, Any]]:
        """Get context for clarification response"""
        return self.last_clarification_context
    
    def set_clarification_context(self, context: Dict[str, Any]):
        """Set context for pending clarification"""
        self.last_clarification_context = context
    
    def clear_clarification_context(self):
        """Clear pending clarification context"""
        self.last_clarification_context = None
    
    def set_last_meeting_context(self, meeting_info: Dict[str, Any]):
        """Set the last meeting context for follow-ups"""
        self.last_meeting_context = meeting_info
    
    def get_last_meeting_context(self) -> Optional[Dict[str, Any]]:
        """Get the last meeting context"""
        return self.last_meeting_context

class ReasoningEngine(BaseComponent):
    """Main reasoning engine for processing F1 queries"""
    
    def __init__(self, config, db_config: DatabaseConfig, tools: Dict[str, Any], http_client):
        super().__init__(config, db_config)
        self.tools = tools
        self.http_client = http_client
        self.intent_analyzer = IntentAnalyzer(config, db_config)
        self.context_manager = ContextManager(config, db_config)

        self.logger.info(f"🔍 Config openai_api_key type: {type(config.openai_api_key)}")
        self.logger.info(f"🔍 Config openai_api_key value: {config.openai_api_key}")
        
        # Try to get from environment directly as fallback
        env_key = os.getenv('OPENAI_API_KEY')
        if not config.openai_api_key and env_key:
            self.logger.info("🔧 Using environment variable as fallback")
            api_key = env_key
        else:
            api_key = config.openai_api_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=api_key
        )

    def reason_and_answer(self, user_query: str) -> str:
        """Main reasoning method that analyzes query and generates response"""
        try:
            # Analyze the query
            query_analysis = self.intent_analyzer.analyze_query_intent(user_query, self.context_manager.conversation_history)
            
            # Check if clarification is needed
            clarification = self._check_for_clarification_needed(user_query, query_analysis)
            if clarification:
                return clarification
            
            # Execute tools based on query analysis
            tool_results = self._execute_tools_for_query(user_query, query_analysis)
            
            # Store visualization data if present for frontend
            if "visualization" in tool_results and tool_results["visualization"].get("success"):
                # Store visualization data for the frontend
                # We need to access the pipeline instance to store the visualization
                # This will be handled by the pipeline calling this method
                pass
            
            # Generate intelligent summary
            if "error" in tool_results:
                response = f"I apologize, but I encountered an error: {tool_results['error']}"
            else:
                response = self._generate_intelligent_summary(user_query, query_analysis, tool_results)
            
            # Add to conversation history
            self.context_manager.add_to_history(user_query, response)
            
            # Return both response and visualization data
            # The pipeline will handle storing the visualization data
            return response, tool_results.get("visualization")
            
        except Exception as e:
            self.logger.error(f"❌ Error in reasoning: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}", None
    
    def _check_for_clarification_needed(self, user_query: str, query_analysis: Dict[str, Any]) -> Optional[str]:
        """Check if clarification is needed with enhanced context awareness"""
        
        # Check if meeting is identified
        if not query_analysis["meeting_info"]["name"]:
            # Enhanced follow-up pattern detection (like your notebook)
            query_lower = user_query.lower()
            follow_up_patterns = [
                "what about", "how about", "and", "also", "who got", "who finished",
                "compare", "versus", "vs", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
                "main race", "grand prix", "sprint", "sprint race", "qualifying"  # ← Added these
            ]
            
            is_follow_up = any(pattern in query_lower for pattern in follow_up_patterns)
            
            if is_follow_up and self.context_manager.get_last_meeting_context():
                meeting_info = self.context_manager.get_last_meeting_context()
                self.logger.info(f"🔄 Using stored meeting context: {meeting_info['name']}")
                query_analysis["meeting_info"] = meeting_info
                return None  # No clarification needed, we have context
            elif self.context_manager.conversation_history:
                # Look for race mentions in recent conversation history
                for entry in reversed(self.context_manager.conversation_history[-3:]):
                    previous_meeting = self.intent_analyzer._extract_meeting_info(entry["message"])
                    if previous_meeting["name"]:
                        self.logger.info(f"�� Using context from previous query: {previous_meeting['name']}")
                        query_analysis["meeting_info"] = previous_meeting
                        return None  # No clarification needed, we have context
            
            return ("I'd be happy to help! However, I couldn't identify which race you're asking about. "
                   "Could you please specify the race? For example: 'Bahrain Grand Prix', 'Miami Grand Prix', "
                   "'Chinese Grand Prix', etc.")
        
        # Check for ambiguous session types in weekends with both Sprint and Race
        query_lower = user_query.lower()
        meeting_name = query_analysis["meeting_info"]["name"]
        
        # Skip clarification if we just provided one (like your notebook)
        if self.context_manager.conversation_history:
            last_response = self.context_manager.conversation_history[-1]["response"]
            if "Which one would you like to know about?" in last_response:
                return None  # Don't ask for clarification again
        
        # Check if this weekend has both Sprint and Race
        weekend_has_sprint = self._check_if_weekend_has_sprint(query_analysis["meeting_info"])
        
        if weekend_has_sprint and query_analysis["session_type"] == "Race":
            # Check if query is ambiguous (like your notebook)
            ambiguous_terms = ["who won", "winner", "results", "race"]
            specific_terms = ["main race", "grand prix", "gp", "sprint race", "sprint"]
            
            has_ambiguous = any(term in query_lower for term in ambiguous_terms)
            has_specific = any(term in query_lower for term in specific_terms)
            
            if has_ambiguous and not has_specific:
                # Store the clarification context
                self.context_manager.set_clarification_context({
                    "meeting_info": query_analysis["meeting_info"],
                    "query_type": query_analysis["query_type"],
                    "drivers": query_analysis["drivers"]
                })
                
                return (f"The {meeting_name} weekend has both a Sprint Race and the main Grand Prix. "
                       f"Which one would you like to know about?\n"
                       f"• Main race (Grand Prix)\n"
                       f"• Sprint race\n"
                       f"Please clarify which race you're interested in!")
        
        return None
    
    def _check_if_weekend_has_sprint(self, meeting_info: Dict[str, Any]) -> bool:
        """Check if a weekend has sprint sessions"""
        try:
            if not meeting_info["name"]:
                return False
                
            # Get meeting key first
            meeting_result = self.http_client.call_tool("get_meeting_key", {
                "event_name": meeting_info["name"],
                "year": meeting_info["year"]
            })
            
            if not meeting_result.get("success"):
                return False
            
            meeting_key = meeting_result.get("meeting_key")
            
            # Check available sessions
            try:
                sprint_result = self.http_client.call_tool("get_session_key", {
                    "meeting_key": meeting_key,
                    "session_type": "Sprint"
                })
                return sprint_result.get("success", False)
            except:
                return False
                
        except Exception as e:
            self.logger.warning(f"Could not check sprint availability: {e}")
            return False
    
    def _execute_tools_for_query(self, user_query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tools based on query analysis with proper session detection"""
        tool_results = {}
        sql_queries = []
        
        try:
            meeting_info = query_analysis["meeting_info"]
            session_type = query_analysis["session_type"]
            
            # Step 1: Get meeting key
            meeting_result = self.http_client.call_tool("get_meeting_key", {
                "event_name": meeting_info["name"],
                "year": meeting_info["year"]
            })
            if "sql_query" in meeting_result:
                sql_queries.append({"tool": "get_meeting_key", "query": meeting_result["sql_query"], "params": meeting_result.get("sql_params", {})})
            if not meeting_result.get("success"):
                return {"error": f"Failed to get meeting key: {meeting_result.get('error')}"}
            meeting_key = meeting_result.get("meeting_key")
            tool_results["meeting_info"] = meeting_result
            
            # Step 2: Get session key with proper session type
            session_result = self.http_client.call_tool("get_session_key", {
                "meeting_key": meeting_key,
                "session_type": session_type
            })
            if "sql_query" in session_result:
                sql_queries.append({"tool": "get_session_key", "query": session_result["sql_query"], "params": session_result.get("sql_params", {})})
            if not session_result.get("success"):
                # Try alternative session types if the requested one fails
                self.logger.warning(f"⚠️ {session_type} session not found, trying alternatives...")
                
                if session_type == "Sprint":
                    # Try Race instead
                    session_result = self.http_client.call_tool("get_session_key", {
                        "meeting_key": meeting_key,
                        "session_type": "Race"
                    })
                    if session_result.get("success"):
                        self.logger.info("📝 Note: Using Race session as Sprint not available")
                
                if not session_result.get("success"):
                    return {"error": f"No {session_type} session found for this race weekend"}
            
            session_key = session_result.get("session_key")
            tool_results["session_info"] = session_result
            
            # Step 3: Execute specific tools based on query type
            query_type = query_analysis["query_type"]
            drivers = query_analysis["drivers"]
            
            # Extract team names for all query types
            teams = self._extract_team_names(user_query)
            
            if query_type == "visualization":
                # Determine visualization type based on query content
                viz_type = self._determine_visualization_type(user_query, query_analysis)
                
                # Prepare filters - handle multiple drivers/teams
                driver_filter = None
                team_filter = None
                
                if drivers:
                    if len(drivers) == 1:
                        driver_filter = drivers[0]
                    else:
                        # Multiple drivers - join with commas
                        driver_filter = ", ".join(drivers)
                
                if teams:
                    if len(teams) == 1:
                        team_filter = teams[0]
                    else:
                        # Multiple teams - join with commas
                        team_filter = ", ".join(teams)
                
                self.logger.info(f"🔍 Visualization - Type: {viz_type}, Driver: {driver_filter}, Team: {team_filter}")
                
                if viz_type == "lap_time_progression":
                    tool_results["visualization"] = self.http_client.call_tool("create_lap_time_progression", {
                        "session_key": session_key,
                        "driver_filter": driver_filter,
                        "team_filter": team_filter
                    })
                    if "sql_query" in tool_results["visualization"]:
                        sql_queries.append({"tool": "create_lap_time_progression", "query": tool_results["visualization"]["sql_query"], "params": tool_results["visualization"].get("sql_params", {})})
                elif viz_type == "position_progression":
                    tool_results["visualization"] = self.http_client.call_tool("create_position_progression", {
                        "session_key": session_key,
                        "driver_filter": driver_filter,
                        "team_filter": team_filter
                    })
                    if "sql_query" in tool_results["visualization"]:
                        sql_queries.append({"tool": "create_position_progression", "query": tool_results["visualization"]["sql_query"], "params": tool_results["visualization"].get("sql_params", {})})
                elif viz_type == "sector_analysis":
                    tool_results["visualization"] = self.http_client.call_tool("create_sector_analysis", {
                        "session_key": session_key,
                        "driver_filter": driver_filter,
                        "team_filter": team_filter
                    })
                    if "sql_query" in tool_results["visualization"]:
                        sql_queries.append({"tool": "create_sector_analysis", "query": tool_results["visualization"]["sql_query"], "params": tool_results["visualization"].get("sql_params", {})})
                elif viz_type == "pit_stop_analysis":
                    # Determine analysis type based on user query
                    analysis_type = self._determine_pit_stop_analysis_type(user_query, query_analysis)
                    
                    tool_results["visualization"] = self.http_client.call_tool("create_pit_stop_analysis", {
                        "session_key": session_key,
                        "driver_filter": driver_filter,
                        "team_filter": team_filter,
                        "analysis_type": analysis_type
                    })
                    if "sql_query" in tool_results["visualization"]:
                        sql_queries.append({"tool": "create_pit_stop_analysis", "query": tool_results["visualization"]["sql_query"], "params": tool_results["visualization"].get("sql_params", {})})
                elif viz_type == "tire_strategy":
                    tool_results["visualization"] = self.http_client.call_tool("create_tire_strategy", {
                        "session_key": session_key,
                        "driver_filter": driver_filter,
                        "team_filter": team_filter
                    })
                    if "sql_query" in tool_results["visualization"]:
                        sql_queries.append({"tool": "create_tire_strategy", "query": tool_results["visualization"]["sql_query"], "params": tool_results["visualization"].get("sql_params", {})})
                else:
                    # Default to lap time progression
                    tool_results["visualization"] = self.http_client.call_tool("create_lap_time_progression", {
                        "session_key": session_key,
                        "driver_filter": driver_filter,
                        "team_filter": team_filter
                    })
                    if "sql_query" in tool_results["visualization"]:
                        sql_queries.append({"tool": "create_lap_time_progression", "query": tool_results["visualization"]["sql_query"], "params": tool_results["visualization"].get("sql_params", {})})
                        
            elif query_type == "comparison":
                # Check if it's a team comparison
                if len(teams) >= 2:
                    # Pass all teams to the tool
                    tool_results["team_comparison"] = self.http_client.call_tool("compare_teams", {
                        "session_key": session_key,
                        "teams": teams,  # Pass the full list of teams
                        "comparison_metrics": ["all"]
                    })
                    if "sql_query" in tool_results["team_comparison"]:
                        sql_queries.append({"tool": "compare_teams", "query": tool_results["team_comparison"]["sql_query"], "params": tool_results["team_comparison"].get("sql_params", {})})
                elif len(drivers) >= 2:
                    tool_results["driver_comparison"] = self.http_client.call_tool("compare_drivers", {
                        "session_key": session_key,
                        "drivers": drivers,
                        "comparison_metrics": ["all"]
                    })
                    if "sql_query" in tool_results["driver_comparison"]:
                        sql_queries.append({"tool": "compare_drivers", "query": tool_results["driver_comparison"]["sql_query"], "params": tool_results["driver_comparison"].get("sql_params", {})})
                else:
                    tool_results["error"] = f"Need two or more drivers or teams to compare. Found drivers: {drivers}, teams: {teams}"
            
            elif query_type == "driver_performance":
                if len(drivers) >= 1:
                    tool_results["driver_performance"] = self.http_client.call_tool("get_driver_performance", {
                        "session_key": session_key,
                        "driver_name": drivers[0],
                        "metrics": ["all"]
                    })
                    if "sql_query" in tool_results["driver_performance"]:
                        sql_queries.append({"tool": "get_driver_performance", "query": tool_results["driver_performance"]["sql_query"], "params": tool_results["driver_performance"].get("sql_params", {})})
                else:
                    tool_results["error"] = "Need to specify a driver for performance analysis"

            elif query_type == "team_performance":
                # Extract team name from query
                if teams:
                    tool_results["team_performance"] = self.http_client.call_tool("get_team_performance", {
                        "session_key": session_key,
                        "team_name": teams[0]  # Changed from "team_filter" to "team_name"
                    })
                    if "sql_query" in tool_results["team_performance"]:
                        sql_queries.append({"tool": "get_team_performance", "query": tool_results["team_performance"]["sql_query"], "params": tool_results["team_performance"].get("sql_params", {})})
                else:
                    tool_results["error"] = "Need to specify a team for performance analysis"
                        
            elif query_type == "fastest_lap":
                tool_results["fastest_lap"] = self.http_client.call_tool("get_fastest_lap", {
                    "session_key": session_key
                })
                if "sql_query" in tool_results["fastest_lap"]:
                    sql_queries.append({"tool": "get_fastest_lap", "query": tool_results["fastest_lap"]["sql_query"], "params": tool_results["fastest_lap"].get("sql_params", {})})
            
            elif query_type == "tire_strategy":
                # Extract team or driver filter
                tool_results["tire_strategy"] = self.http_client.call_tool("get_tire_strategy", {
                    "session_key": session_key,
                    "team_filter": teams[0] if teams else None,
                    "driver_filter": drivers[0] if drivers else None,
                    "strategy_type": "all"
                })
                if "sql_query" in tool_results["tire_strategy"]:
                    sql_queries.append({"tool": "get_tire_strategy", "query": tool_results["tire_strategy"]["sql_query"], "params": tool_results["tire_strategy"].get("sql_params", {})})

            elif query_type == "pit_strategy":
                tool_results["pit_stop_analysis"] = self.http_client.call_tool("get_pit_stop_analysis", {
                    "session_key": session_key,
                    "analysis_type": "all"
                })
                if "sql_query" in tool_results["pit_stop_analysis"]:
                    sql_queries.append({"tool": "get_pit_stop_analysis", "query": tool_results["pit_stop_analysis"]["sql_query"], "params": tool_results["pit_stop_analysis"].get("sql_params", {})})
            
            elif query_type == "incident_investigation":
                lap_number = query_analysis.get("lap_number")
                
                # Determine analysis type based on query
                if drivers and lap_number:
                    analysis_type = "driver_specific_lap"
                elif drivers:
                    analysis_type = "driver"
                elif lap_number:
                    analysis_type = "specific_lap"
                else:
                    analysis_type = "summary"
                
                tool_results["incident_analysis"] = self.http_client.call_tool("investigate_incident", {
                    "session_key": session_key,
                    "driver_name": drivers[0] if drivers else None,
                    "lap_number": lap_number,
                    "analysis_type": analysis_type
                })
                if "sql_query" in tool_results["incident_analysis"]:
                    sql_queries.append({"tool": "investigate_incident", "query": tool_results["incident_analysis"]["sql_query"], "params": tool_results["incident_analysis"].get("sql_params", {})})
            
            elif query_type == "sector_analysis":
                # Determine sector analysis type based on query
                query_lower = user_query.lower()
                if any(word in query_lower for word in ["consistency", "consistent", "reliability"]):
                    sector_analysis_type = "consistency"
                elif any(word in query_lower for word in ["compare", "comparison", "vs", "versus"]):
                    sector_analysis_type = "comparison"
                elif any(word in query_lower for word in ["best", "fastest", "quickest"]):
                    sector_analysis_type = "best_sectors"
                else:
                    sector_analysis_type = "all"
                
                tool_results["sector_analysis"] = self.http_client.call_tool("get_sector_analysis", {
                    "session_key": session_key,
                    "driver_filter": drivers[0] if drivers else None,
                    "team_filter": teams[0] if teams else None,
                    "sector_analysis_type": sector_analysis_type
                })
                if "sql_query" in tool_results["sector_analysis"]:
                    sql_queries.append({"tool": "get_sector_analysis", "query": tool_results["sector_analysis"]["sql_query"], "params": tool_results["sector_analysis"].get("sql_params", {})})
            
            elif query_type == "qualifying_results":
                tool_results["qualifying_results"] = self.http_client.call_tool("get_qualifying_results", {
                    "session_key": session_key,
                    "result_type": "top_10"
                })
                if "sql_query" in tool_results["qualifying_results"]:
                    sql_queries.append({"tool": "get_qualifying_results", "query": tool_results["qualifying_results"]["sql_query"], "params": tool_results["qualifying_results"].get("sql_params", {})})
            
            else:  # race_results
                # Extract number from query to determine result type
                result_type = self._determine_race_result_type(user_query)
                
                tool_results["race_results"] = self.http_client.call_tool("get_race_results", {
                    "session_key": session_key,
                    "result_type": result_type,
                    "include_lap_times": True
                })
                if "sql_query" in tool_results["race_results"]:
                    sql_queries.append({"tool": "get_race_results", "query": tool_results["race_results"]["sql_query"], "params": tool_results["race_results"].get("sql_params", {})})
            
            tool_results["sql_queries"] = sql_queries
            return tool_results
        
        except Exception as e:
            tool_results["error"] = str(e)
            self.logger.error(f"❌ Error executing tools: {e}")
            return tool_results
    

    def _determine_pit_stop_analysis_type(self, user_query: str, query_analysis: Dict[str, Any]) -> str:
        """Determine the type of pit stop analysis based on user query"""
        query_lower = user_query.lower()
        
        # Extract drivers and teams from query analysis
        drivers = query_analysis.get("drivers", [])
        teams = query_analysis.get("teams", [])
        
        # Check for comparison keywords
        comparison_keywords = ["compare", "vs", "versus", "difference", "who was better", "which driver was faster"]
        for keyword in comparison_keywords:
            if keyword in query_lower:
                return "comparison"
        
        # Check for comprehensive analysis keywords
        comprehensive_keywords = ["comprehensive", "detailed", "full", "complete", "all", "everything", "thorough"]
        for keyword in comprehensive_keywords:
            if keyword in query_lower:
                return "comprehensive"
        
        # Check for simple analysis keywords
        simple_keywords = ["simple", "basic", "quick", "overview", "summary"]
        for keyword in simple_keywords:
            if keyword in query_lower:
                return "simple"
        
        # Determine based on number of drivers/teams
        if len(drivers) > 1 or len(teams) > 1:
            return "comparison"
        elif len(drivers) == 1 and len(teams) == 0:
            return "simple"
        elif len(teams) == 1 and len(drivers) == 0:
            return "simple"
        elif len(drivers) == 1 and len(teams) == 1:
            return "simple"
        else:
            # Default to comprehensive for general queries
            return "comprehensive"
    

    def _determine_visualization_type(self, user_query: str, query_analysis: Dict[str, Any]) -> str:
        """Determine the type of visualization to create based on user query content"""
        query_lower = user_query.lower()
        
        # Check for specific visualization types with priority order
        
        # 1. Position-related visualizations
        position_keywords = [
            "position", "positions", "grid", "order", "finishing position", 
            "race order", "grid position", "starting position", "final position",
            "position changes", "position progression", "how did positions change"
        ]
        if any(keyword in query_lower for keyword in position_keywords):
            self.logger.info(f"✅ Detected position progression visualization")
            return "position_progression"
        
        # 2. Sector-related visualizations
        sector_keywords = [
            "sector", "sectors", "sector 1", "sector 2", "sector 3", 
            "first sector", "second sector", "third sector", "sector times",
            "sector analysis", "sector performance", "sector breakdown",
            "sector strengths", "sector weaknesses", "sector comparison"
        ]
        if any(keyword in query_lower for keyword in sector_keywords):
            self.logger.info(f"✅ Detected sector analysis visualization")
            return "sector_analysis"
        
        # 3. Pit stop-related visualizations
        pit_stop_keywords = [
            "pit", "pitstop", "pit stop", "pitstop", "stops", "pit stops",
            "pit stop analysis", "pit stop strategy", "pit stop times",
            "fastest pit stop", "pit stop duration", "pit stop performance",
            "when did they pit", "how many stops", "pit stop timing"
        ]
        if any(keyword in query_lower for keyword in pit_stop_keywords):
            self.logger.info(f"✅ Detected pit stop analysis visualization")
            return "pit_stop_analysis"
        
        # 4. Tire strategy-related visualizations
        tire_strategy_keywords = [
            "tire", "tyre", "compound", "strategy", "stint", "stints",
            "tire strategy", "tyre strategy", "tire compounds", "tyre compounds",
            "tire management", "tyre management", "compound strategy",
            "stint analysis", "tire wear", "tyre wear", "compound usage"
        ]
        if any(keyword in query_lower for keyword in tire_strategy_keywords):
            return "tire_strategy"
        
        # 5. Lap time-related visualizations
        lap_time_keywords = [
            "lap time", "lap times", "lap duration", "lap progression",
            "lap analysis", "lap performance", "lap chart", "lap graph",
            "time progression", "lap by lap", "lap comparison"
        ]
        if any(keyword in query_lower for keyword in lap_time_keywords):
            return "lap_time_progression"
        
        # 6. General visualization requests - try to infer from context
        general_viz_keywords = [
            "show me a graph", "create a chart", "visualize", "plot", "graph",
            "show me the data", "display", "chart", "visualization", "graphical",
            "show me a visualization", "create a plot", "show the trend",
            "plot the data", "graph the results", "show me the progression",
            "visualize the data", "create a diagram", "show me the analysis",
        ]
        if any(keyword in query_lower for keyword in general_viz_keywords):
            # For general requests, check the query type to determine best visualization
            query_type = query_analysis.get("query_type", "")
            
            if query_type == "comparison":
                # For comparisons, lap time progression is usually most useful
                return "lap_time_progression"
            elif query_type == "driver_performance" or query_type == "team_performance":
                # For performance analysis, sector analysis might be more insightful
                return "sector_analysis"
            elif query_type == "race_results":
                # For race results, position progression is most relevant
                return "position_progression"
            else:
                # Default to lap time progression for general requests
                return "lap_time_progression"
        
        # 7. If no specific keywords found, check for driver/team mentions
        drivers = query_analysis.get("drivers", [])
        teams = self._extract_team_names(user_query)
        
        if drivers or teams:
            # If specific drivers/teams mentioned, lap time progression is usually best
            return "lap_time_progression"
        
        # 8. Final fallback - default to lap time progression
        return "lap_time_progression"
    
        
    def _extract_team_names(self, query: str) -> List[str]:
        """Extract team names from query"""
        team_mappings = {
            "mercedes": "Mercedes",
            "ferrari": "Ferrari", 
            "mclaren": "McLaren",
            "red bull": "Red Bull Racing",
            "redbull": "Red Bull Racing",
            "red bull racing": "Red Bull Racing",
            "williams": "Williams",
            "alpine": "Alpine",
            "aston martin": "Aston Martin",
            "astonmartin": "Aston Martin",
            "haas": "Haas F1 Team",
            "haas f1": "Haas F1 Team",
            "rb": "Racing Bulls",
            "bulls": "Racing Bulls",
            "racing bulls": "Racing Bulls",
            "sauber": "Sauber",
            "stake": "Stake F1 Team",
            "stake f1": "Stake F1 Team"
        }
        
        query_lower = query.lower()
        found_teams = []
        
        for team_alias, team_name in team_mappings.items():
            if team_alias in query_lower:
                if team_name not in found_teams:
                    found_teams.append(team_name)
        
        return found_teams
    
    def _determine_race_result_type(self, query: str) -> str:
        """Determine the result type based on the user query"""
        query_lower = query.lower()
        
        # Check for specific numbers
        import re
        numbers = re.findall(r'\d+', query_lower)
        
        if numbers:
            num = int(numbers[0])
            if num == 1:
                return "winner_only"
            elif num == 3:
                return "podium"
            else:
                return f"top_{num}"  # Dynamic result type
        
        # Check for keywords
        if any(word in query_lower for word in ["winner", "won", "first"]):
            return "winner_only"
        elif any(word in query_lower for word in ["podium", "top 3", "first 3"]):
            return "podium"
        else:
            return "podium" 
        
    def _generate_intelligent_summary(self, user_query: str, query_analysis: Dict[str, Any], tool_results: Dict[str, Any]) -> str:
        """Generate an intelligent text summary using LLM"""
        
        if "error" in tool_results:
            return f"I apologize, but I encountered an error: {tool_results['error']}"
        
        # Get conversation context
        conversation_context = self._get_conversation_context()
        
        # Check if visualization was generated and get its type safely
        has_visualization = "visualization" in tool_results and tool_results["visualization"].get("success")
        viz_type = "data visualization"
        
        # Create a filtered version of tool_results for LLM (exclude visualization data)
        filtered_tool_results = {}
        for key, value in tool_results.items():
            if key == "visualization":
                # Only include metadata, not the actual image data
                if has_visualization:
                    viz_data = value
                    filtered_tool_results[key] = {
                        "success": viz_data.get("success"),
                        "visualization_type": viz_data.get("visualization_type", "data visualization"),
                        "filename": viz_data.get("filename"),
                        "session_key": viz_data.get("session_key"),
                        "total_laps": viz_data.get("total_laps"),
                        "total_drivers": viz_data.get("total_drivers"),
                        "drivers_included": viz_data.get("drivers_included", [])[:5]  # Limit to first 5 drivers
                    }
            else:
                filtered_tool_results[key] = value
        
        # Determine query type and use appropriate prompt
        query_type = query_analysis.get("query_type", "")
        
        if has_visualization or query_type == "visualization":
            summary_prompt = self._get_visualization_prompt(user_query, query_analysis, filtered_tool_results, conversation_context, viz_type)
        else:
            summary_prompt = self._get_text_query_prompt(user_query, query_analysis, filtered_tool_results, conversation_context)
        
        try:
            messages = [SystemMessage(content=summary_prompt)]
            response = self.llm.invoke(messages)
            answer = response.content
            
            return answer
            
        except Exception as e:
            self.logger.error(f"❌ Error generating summary: {e}")
            # Fallback to basic summary
            return self._generate_basic_summary(user_query, query_analysis, tool_results)

    def _get_visualization_prompt(self, user_query: str, query_analysis: Dict[str, Any], filtered_tool_results: Dict[str, Any], conversation_context: str, viz_type: str) -> str:
        """Generate prompt specifically for visualization queries"""
        return f"""{conversation_context}You are an expert F1 analyst. Based on the following data, provide a friendly and informative summary about the visualization you've created.

    User Query: {user_query}
    Race: {query_analysis["meeting_info"]["name"]} {query_analysis["meeting_info"]["year"]}
    Session: {query_analysis["session_type"]}

    Data Retrieved:
    {json.dumps(filtered_tool_results, indent=2)}

    I have generated a {viz_type} for you to view below.

    VISUALIZATION RESPONSE REQUIREMENTS:
    1. Start with a friendly, brief introduction about what you've created
    2. Mention the specific race and session
    3. Explain what the visualization shows and what insights can be gained
    4. Mention that graph analysis capabilities will be available soon
    5. Keep it concise and to the point

    FORMATTING FOR VISUALIZATIONS:
    - Be friendly but concise
    - Do NOT list individual driver names unless specifically asked
    - Focus on what the visualization shows, not who is included
    - Use proper line breaks between sections
    - Keep the tone engaging but factual

    Example format:
    I've created a lap time progression visualization for the Australian Grand Prix 2025 Race that shows how each driver's performance evolved throughout the session.

    The visualization below displays lap time progression for all drivers over the course of the race. You can see how lap times varied throughout the session, including pit stops and any incidents that affected performance.

    Graph analysis capabilities will be available soon to help you dive deeper into the performance data.

    Write your response in a friendly, conversational tone with proper line breaks and formatting.
    """

    def _get_text_query_prompt(self, user_query: str, query_analysis: Dict[str, Any], filtered_tool_results: Dict[str, Any], conversation_context: str) -> str:
        """Generate prompt specifically for text-based queries"""
        return f"""{conversation_context}You are an expert F1 analyst. Based on the following data, provide a friendly and informative summary that directly answers the user's question.

    User Query: {user_query}
    Race: {query_analysis["meeting_info"]["name"]} {query_analysis["meeting_info"]["year"]}
    Session: {query_analysis["session_type"]}

    Data Retrieved:
    {json.dumps(filtered_tool_results, indent=2)}

    TEXT QUERY RESPONSE REQUIREMENTS:
    1. Directly answer the user's question in a friendly, conversational tone
    2. Include specific numbers, times, and positions
    3. Use clear, factual language but be engaging
    4. Convert lap times from seconds to MM:SS.mmm format (e.g., 93.614 seconds becomes 01:33.614)
    5. Mention significant gaps only if noteworthy
    6. Provide context and insights about the results

    FORMATTING FOR TEXT QUERIES:
    - Be friendly and conversational while staying factual
    - Avoid overly dramatic phrases like "thrilling", "spectacular", "vibrant skies", "edge of their seats"
    - Focus on facts: who, what times, what positions
    - When displaying lap times, use MM:SS.mmm format
    - If multiple fastest laps are provided, list them in order as a list with each entry as a new line from fastest to slowest
    - Use proper line breaks and spacing for readability
    - Provide additional context and insights when relevant

    IMPORTANT: When displaying lap times, use the format MM:SS.mmm (e.g., 01:33.614, 01:30.000, 01:35.123). Do NOT mix words and numbers for times.

    Example format:
    For the Australian Grand Prix 2025 Race, here are the fastest lap times:

    1. Charles Leclerc (Ferrari) - 01:24.567
    2. Carlos Sainz (Ferrari) - 01:24.789
    3. Esteban Ocon (Alpine) - 01:25.123

    Charles Leclerc set the fastest lap of the session, with his Ferrari teammate Carlos Sainz close behind. The Alpine of Esteban Ocon rounded out the top three, showing strong pace throughout the session.

    Write your response in a friendly, conversational tone with proper line breaks and formatting.
    """
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation context for the LLM (like your notebook)"""
        if not self.context_manager.conversation_history:
            return ""
        
        context = "Recent conversation context:\n"
        for entry in self.context_manager.conversation_history[-2:]:  # Last 2 exchanges
            context += f"User: {entry['message']}\n"
            # Truncate response to avoid too much context
            response_preview = entry['response'][:200] + "..." if len(entry['response']) > 200 else entry['response']
            context += f"Assistant: {response_preview}\n\n"
        
        return context
    
    def _generate_basic_summary(self, user_query: str, query_analysis: Dict[str, Any], tool_results: Dict[str, Any]) -> str:
        """Generate a basic summary if LLM fails"""
        
        summary = f"Here's what I found for the {query_analysis['meeting_info']['name']} {query_analysis['meeting_info']['year']} {query_analysis['session_type']}:\n\n"
        
        if "driver_comparison" in tool_results and tool_results["driver_comparison"].get("success"):
            comparison_data = tool_results["driver_comparison"]["comparison"]
            summary += f"Driver Comparison:\n"
            for driver_data in comparison_data:
                summary += f"• {driver_data['driver']} ({driver_data['team']}): {driver_data.get('total_laps', 'N/A')} laps, best: {driver_data.get('best_lap', 'N/A')}s\n"
        
        elif "race_results" in tool_results and tool_results["race_results"].get("success"):
            results = tool_results["race_results"]["results"]
            summary += "Race Results:\n"
            for i, result in enumerate(results[:3], 1):
                summary += f"{i}. {result['driver']} ({result['team']}) - Position {result['position']}\n"
        
        return summary