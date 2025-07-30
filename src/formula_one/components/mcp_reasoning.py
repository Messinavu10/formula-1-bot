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
        """Analyze user query to determine intent and extract relevant information"""
        
        # Add debugging
        self.logger.info(f"🔍 Analyzing query intent for: '{user_query}'")
        
        query_lower = user_query.lower()
        
        # Extract meeting information
        meeting_info = self._extract_meeting_info(user_query)
        
        # Determine session type
        session_type = self._determine_session_type(query_lower)
        
        # Check if this is a generic query that should inherit from conversation history
        generic_queries = ["summary", "show me", "tell me", "what about", "and", "also"]
        should_inherit = any(generic in query_lower for generic in generic_queries)
        
        self.logger.info(f"🔍 Should inherit: {should_inherit}")
        self.logger.info(f"🔍 Conversation history: {conversation_history}")
        
        if should_inherit and conversation_history:
            # Look for specific query types in recent conversation history
            for i, entry in enumerate(reversed(conversation_history[-3:])):
                self.logger.info(f"🔍 Checking history entry {i}: {entry}")
                previous_query_type = self._determine_query_type(entry["message"].lower())
                self.logger.info(f"🔍 Previous query type: {previous_query_type}")
                if previous_query_type in ["tire_strategy", "pit_strategy", "sector_analysis", "comparison","position_analysis"]:
                    self.logger.info(f"🔍 Using query type from conversation history: {previous_query_type}")
                    query_type = previous_query_type
                    break
            else:
                # If no specific query type found in history, determine normally
                self.logger.info(f"🔍 No specific query type found in history, determining normally")
                query_type = self._determine_query_type(query_lower)
        else:
            # Determine query type normally
            self.logger.info(f"🔍 Determining query type normally")
            query_type = self._determine_query_type(query_lower)
        
        # Add debugging
        self.logger.info(f"🔍 Query type determined: {query_type}")
        
        # Extract driver names from current query
        drivers = self._extract_driver_names(user_query)
        
        # Extract team names from current query
        teams = self._extract_team_names(user_query)
        
        # If no teams/drivers found in current query, try to get from conversation history
        if not teams and not drivers and conversation_history:
            # Look for teams/drivers in recent conversation history
            for entry in reversed(conversation_history[-3:]):
                previous_teams = self._extract_team_names(entry["message"])
                previous_drivers = self._extract_driver_names(entry["message"])
                
                if previous_teams:
                    self.logger.info(f"🔍 Using teams from conversation history: {previous_teams}")
                    teams = previous_teams
                    break
                elif previous_drivers:
                    self.logger.info(f"�� Using drivers from conversation history: {previous_drivers}")
                    drivers = previous_drivers
                    break
        
        # Extract lap number if present
        lap_number = self._extract_lap_number(user_query)
        
        # Get context from conversation history
        context = self._get_context_from_history(conversation_history)
        
        # Combine all analysis results
        analysis_result = {
            "query_type": query_type,  # Make sure this is included
            "meeting_info": meeting_info,
            "session_type": session_type,
            "drivers": drivers,
            "teams": teams,  # Add teams to the analysis result
            "lap_number": lap_number,
            "context": context
        }
        
        # Add debugging
        self.logger.info(f"�� Analysis result: {analysis_result}")
        
        return analysis_result
    
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
        
        # Add debugging
        self.logger.info(f"🔍 Extracting teams from: '{query_lower}'")
        
        for team_alias, team_name in team_mappings.items():
            if team_alias in query_lower:
                self.logger.info(f"✅ Found team alias '{team_alias}' -> '{team_name}'")
                if team_name not in found_teams:
                    found_teams.append(team_name)
        
        self.logger.info(f"🔍 Final extracted teams: {found_teams}")
        return found_teams

    def _determine_query_type(self, query_lower: str) -> str:
        """Determine query type using semantic similarity with priority logic"""
        
        # Add debugging
        self.logger.info(f"🔍 Determining query type for: '{query_lower}'")
        
        # First, check for visualization keywords (highest priority)
        viz_keywords = [
            "show me a graph", "create a chart", "visualize", "plot", "graph",
            "show me the data", "display", "chart", "visualization", "graphical",
            "show me a visualization", "create a plot", "show the trend",
            "plot the data", "graph the results", "show me the progression",
            "visualize the data", "create a diagram", "show me the analysis",
            "show me", "sector chart", "position chart", "lap chart", "tire chart",
            "pit stop chart", "strategy chart"
        ]
        
        for keyword in viz_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched visualization keyword: '{keyword}'")
                return "visualization"
        
        position_progression_keywords = [
        "position progression", "position changes", "position analysis",
        "how did positions change", "position tracking", "position movement",
        "position development", "position evolution", "position history"
        ]
        for keyword in position_progression_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched position progression keyword: '{keyword}'")
                return "position_analysis"
            
        # Second, check for tire strategy keywords (HIGHER PRIORITY than comparison)
        tire_strategy_keywords = [
            "tire strategy", "tire strategies", "tire compound", "tire compounds",
            "tire usage", "tire management", "tire performance", "tire analysis",
            "compound strategy", "compound usage", "compound management",
            "stint", "stints", "stint length", "stint duration",
            "tire comparison", "tire strategy comparison", "compound comparison"
        ]
        for keyword in tire_strategy_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched tire strategy keyword: '{keyword}'")
                return "tire_strategy"
        
        # Second, check for explicit comparison keywords (HIGHER PRIORITY)
        comparison_keywords = ["compare", "vs", "versus", "difference", "who was better", "which driver was faster", "comparison"]
        for keyword in comparison_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched comparison keyword: '{keyword}'")
                return "comparison"
        
        # SPECIAL CASE: Check for sector comparison (should be sector_analysis, not comparison)
        if "sector" in query_lower and any(keyword in query_lower for keyword in ["compare", "comparison", "vs", "versus"]):
            self.logger.info(f"✅ Matched sector comparison - treating as sector_analysis")
            return "sector_analysis"
        
        self.logger.info(f"🔍 No comparison keywords matched")
        
        # Third, check for pit stop keywords (lower priority)
        pit_stop_keywords = [
            "pit stop", "pitstop", "pitted", "when did they pit", "pit strategy",
            "pit stops", "pitstop", "when did", "pit timing", "pit duration",
            "pit stop analysis", "pit stop strategy", "pit stop timing",
            "when drivers pitted", "pit stop times", "pit stop performance"
        ]
        for keyword in pit_stop_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched pit stop keyword: '{keyword}'")
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
        has_lap_number = bool(re.search(r'\blap\s+\d+', query_lower))
        has_incident_keyword = any(keyword in query_lower for keyword in lap_incident_keywords)
        
        if has_lap_number and has_incident_keyword:
            self.logger.info(f"✅ Matched incident investigation")
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
                self.logger.info(f"✅ Matched sector analysis keyword: '{keyword}'")
                return "sector_analysis"
        
        # IMPROVED: Use team extraction for better generalization
        performance_words = ["perform", "performance", "analysis", "stats", "data", "how did", "team performance", "team analysis"]
        has_performance_word = any(word in query_lower for word in performance_words)
        
        # Use the team extraction method to detect any team
        extracted_teams = self._extract_team_names(query_lower)
        has_team_name = len(extracted_teams) > 0
        
        self.logger.info(f"🔍 Team performance check - has_performance_word: {has_performance_word}, has_team_name: {has_team_name}, extracted_teams: {extracted_teams}")
        
        # If query has both team name and performance word, it's likely team performance
        if has_team_name and has_performance_word:
            self.logger.info(f"✅ Matched team performance - teams: {extracted_teams}")
            return "team_performance"
        
        # Also check the original team keywords for backward compatibility
        team_keywords = [
            "team performance", "team analysis", "team stats", "team data",
            "mercedes team", "ferrari team", "mclaren team", "red bull team",
            "team performance for", "team analysis for", "team stats for"
        ]
        for keyword in team_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched team keyword: '{keyword}'")
                return "team_performance"
        
        # Check for driver performance keywords (LOWER PRIORITY)
        driver_keywords = ["driver performance", "driver analysis", "driver stats", "driver data"]
        for keyword in driver_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched driver performance keyword: '{keyword}'")
                return "driver_performance"
        
        # If no explicit keywords, use semantic similarity
        self.logger.info(f"🔍 No explicit keywords matched, using semantic similarity")
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
        if best_score < 0.2:  # Adjust threshold as needed
            self.logger.info(f"🔍 Low similarity score ({best_score:.3f}), defaulting to race_results")
            return "race_results"

        self.logger.info(f"✅ Semantic similarity matched: {best_intent} (score: {best_score:.3f})")
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
        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response
        }
        self.conversation_history.append(entry)
        self.logger.info(f"🔍 Added to conversation history: {entry}")
        self.logger.info(f"�� Total conversation history entries: {len(self.conversation_history)}")
        self.logger.info(f"🔍 Full conversation history: {self.conversation_history}")
    
    def is_clarification_response(self, user_query: str) -> bool:
        """Check if this is a clarification response"""
        if not self.conversation_history:
            return False
        
        last_response = self.conversation_history[-1]["response"]
        
        clarification_patterns = [
            "Would you like me to create a visualization chart",
            "Would you prefer a text summary",
            "Would you like a visualization chart or a text summary",
            "I can help you with that! Would you like me to create",
            "Would you like a visualization or a text summary"
        ]

        return any(pattern in last_response for pattern in clarification_patterns)
    
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
            self.logger.info(f"🧠 Processing query: '{user_query}'")
            self.logger.info(f"🔍 ReasoningEngine instance ID: {id(self)}")
            self.logger.info(f"🔍 ContextManager instance ID: {id(self.context_manager)}")
            self.logger.info(f"🔍 Current conversation history length: {len(self.context_manager.conversation_history)}")
            if self.context_manager.conversation_history:
                self.logger.info(f"🔍 Last conversation entry: {self.context_manager.conversation_history[-1]}")
            
            # Store the current conversation history for analysis (BEFORE adding temp entry)
            current_history = self.context_manager.conversation_history.copy()
            
            # Add the current query to conversation history BEFORE analysis
            # This ensures that subsequent queries can access the previous query
            temp_entry = {
                "timestamp": datetime.now().isoformat(),
                "message": user_query,
                "response": ""  # Empty response for now
            }
            self.context_manager.conversation_history.append(temp_entry)
            
            # Analyze the query using the PREVIOUS history (not including current query)
            query_analysis = self.intent_analyzer.analyze_query_intent(user_query, current_history)
            self.logger.info(f"🧠 Query analysis: {query_analysis}")
            
            # Remove the temporary entry after analysis
            self.context_manager.conversation_history.pop()
            
            self.logger.info(f"🔍 After analysis - checking for ambiguous query")
            # Check for ambiguous queries that need clarification
            ambiguous_response = self._check_for_ambiguous_query(user_query, query_analysis)
            if ambiguous_response:
                self.logger.info(f"🔍 Returning ambiguous response: {ambiguous_response}")
                return ambiguous_response
            
            self.logger.info(f"🔍 After ambiguous check - checking for clarification response")
            # Check for clarification responses
            clarification_response = self._handle_clarification_response(user_query, query_analysis)
            if clarification_response:
                self.logger.info(f"🧠 Clarification response: {clarification_response}")
                return clarification_response
            
            self.logger.info(f"🔍 After clarification check - checking meeting info")
            # Check if meeting is identified
            if not query_analysis["meeting_info"]["name"]:
                self.logger.info(f"🔍 No meeting identified, generating clarification")
                # Use LLM to generate a natural clarification
                clarification_prompt = f"""
    The user asked: "{user_query}"

    I couldn't identify which race they're asking about. Please ask them to specify the race in a friendly, helpful way.
    Suggest some examples like "Bahrain Grand Prix", "Miami Grand Prix", etc.
    """
                
                try:
                    response = self.llm.invoke(clarification_prompt).content
                    return response
                except Exception as e:
                    self.logger.warning(f"Failed to generate meeting clarification with LLM: {e}")
                    return ("I'd be happy to help! However, I couldn't identify which race you're asking about. "
                        "Could you please specify the race? For example: 'Bahrain Grand Prix', 'Miami Grand Prix', "
                        "'Chinese Grand Prix', etc.")
            
            self.logger.info(f"🔍 About to execute tools for query type: {query_analysis['query_type']}")
            # Execute tools based on query analysis
            tool_results = self._execute_tools_for_query(user_query, query_analysis)
            self.logger.info(f"🧠 Tool results: {tool_results}")
            
            # Generate intelligent summary
            if "error" in tool_results:
                response = f"I apologize, but I encountered an error: {tool_results['error']}"
            else:
                response = self._generate_intelligent_summary(user_query, query_analysis, tool_results)
            
            self.logger.info(f"🔍 About to add to conversation history")
            # Add to conversation history
            self.context_manager.add_to_history(user_query, response)
            
            # Debug: Check conversation history after adding
            self.logger.info(f"🔍 After adding to history - length: {len(self.context_manager.conversation_history)}")
            if self.context_manager.conversation_history:
                self.logger.info(f"🔍 After adding - last entry: {self.context_manager.conversation_history[-1]}")
            
            # Return both response and visualization data
            return response, tool_results.get("visualization")
            
        except Exception as e:
            self.logger.error(f"❌ Error in reasoning: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}", None

    def _check_for_ambiguous_query(self, user_query: str, query_analysis: Dict[str, Any]) -> Optional[str]:
        """Check if query is ambiguous and needs clarification about capabilities"""
        query_lower = user_query.lower()
        self.logger.info(f"🔍 Checking for ambiguous query: {user_query}")
        
        # ADD THIS: Skip ambiguous check for specific query types
        if query_analysis.get("query_type") in ["tire_strategy", "pit_strategy", "sector_analysis", "team_performance", "position_analysis"]:
            self.logger.info(f" Skipping ambiguous check for specific query type: {query_analysis.get('query_type')}")
            return None
        
        # First, check for CLEAR visualization keywords (should NOT trigger clarification)
        clear_viz_keywords = [
            "visualize", "visualization", "show me a", "create a", "plot", "graph", "chart",
            "display", "show the", "create a chart", "create a graph", "create a plot",
            "show me the chart", "show me the graph", "show me the plot"
        ]

        has_clear_viz = any(keyword in query_lower for keyword in clear_viz_keywords)
        if has_clear_viz:
            # This is clearly a visualization request, don't ask for clarification
            return None
        
        # Check for CLEAR text analysis keywords (should NOT trigger clarification)
        clear_text_keywords = [
            "summarize", "summary", "tell me", "what were", "give me the data",
            "describe", "explain", "analyze", "report", "details", "comparison", "between",
            "progression", "changes", "movement", "development", "evolution", "history"
        ]
        
        has_clear_text = any(keyword in query_lower for keyword in clear_text_keywords)
        if has_clear_text:
            # This is clearly a text analysis request, don't ask for clarification
            return None
        
        # Now check for ambiguous patterns that could be either visualization or text analysis
        ambiguous_patterns = [
            # Lap time related
            "lap time progression", "lap times", "lap analysis", "lap comparison",
            "lap performance", "lap duration", "lap chart", "lap graph",
            
            # Position related  
            "position progression", "position changes", "position analysis",
            "how did positions change", "position comparison", "position chart",
            
            # Sector related
            "sector analysis", "sector performance", "sector comparison",
            "sector times", "sector breakdown", "sector chart",
            
            # Pit stop related
            "pit stop analysis", "pit stop performance", "pit stop timing",
            "pit stop strategy", "pit stop comparison", "pit stop chart",
            
            # General comparison patterns (only if not clearly visualization/text)
            "between", "versus", "vs", "compare", "comparison",
            
            # General analysis patterns (only if not clearly visualization/text)
            "analysis", "performance", "data", "stats", "statistics"
        ]
        
        has_ambiguous_pattern = any(pattern in query_lower for pattern in ambiguous_patterns)
        
        if has_ambiguous_pattern:
            # Check if query has drivers/teams mentioned
            drivers = query_analysis.get("drivers", [])
            teams = self._extract_team_names(user_query)
            
            if drivers or teams:
                # Store context for when user responds
                self.context_manager.set_clarification_context({
                    "meeting_info": query_analysis["meeting_info"],
                    "session_type": query_analysis["session_type"],
                    "drivers": drivers,
                    "teams": teams,
                    "original_query": user_query
                })
                
                # Use LLM to generate a natural, well-formatted clarification
                clarification_prompt = f"""
    You are a helpful F1 data assistant. The user asked: "{user_query}"

    This query could be interpreted in different ways. I can provide either:
    1. A visualization (chart/graph) of the data
    2. A text summary/analysis of the data

    Please respond in a friendly, conversational way asking the user which they prefer. 
    Make it sound natural and helpful, not robotic. Use proper formatting with line breaks.

    Example response style:
    "I can help you with that! Would you like me to create a visualization chart, or would you prefer a text summary of the data?"
    """
                
                try:
                    response = self.llm.invoke(clarification_prompt).content
                    return response
                except Exception as e:
                    self.logger.warning(f"Failed to generate clarification with LLM: {e}")
                    # Fallback to simple response
                    return "I can help you with that! Would you like me to create a visualization chart, or would you prefer a text summary of the data?"
        
        return None

    def _handle_clarification_response(self, user_query: str, query_analysis: Dict[str, Any]) -> Optional[str]:
        """Handle user's response to clarification questions"""
        query_lower = user_query.lower()
        clarification_context = self.context_manager.get_clarification_context()
        
        if not clarification_context:
            return None
        
        # Use LLM to interpret the user's response
        interpretation_prompt = f"""
    The user was asked about their preference for data analysis. They responded: "{user_query}"

    Based on their response, determine if they want:
    1. A visualization/chart/graph
    2. A text summary/analysis

    Respond with just "visualization" or "text_analysis".

    If their response is unclear, respond with "unclear".
    """
        
        try:
            interpretation = self.llm.invoke(interpretation_prompt).content.strip().lower()
            
            if "visualization" in interpretation:
                # Update query analysis with stored context
                query_analysis.update(clarification_context)
                query_analysis["query_type"] = "visualization"
                
                # Determine visualization type based on original query
                original_query = clarification_context.get("original_query", "").lower()
                
                # More specific detection based on original query content
                if "sector" in original_query:
                    query_analysis["visualization_type"] = "sector_analysis"
                elif "position" in original_query:
                    query_analysis["visualization_type"] = "position_progression"
                elif "pit" in original_query:
                    query_analysis["visualization_type"] = "pit_stop_analysis"
                elif "tire" in original_query or "tyre" in original_query or "compound" in original_query:
                    query_analysis["visualization_type"] = "tire_strategy"
                elif "lap" in original_query or "time" in original_query:
                    query_analysis["visualization_type"] = "lap_time_progression"
                else:
                    # Default to lap time progression for general comparisons
                    query_analysis["visualization_type"] = "lap_time_progression"
                
                self.context_manager.clear_clarification_context()
                return None  # Continue with execution
                
            elif "text_analysis" in interpretation or "text" in interpretation:
                # Update query analysis with stored context
                query_analysis.update(clarification_context)
                
                # Determine appropriate text analysis type based on original query
                original_query = clarification_context.get("original_query", "").lower()
                
                if "compare" in original_query or "vs" in original_query or "versus" in original_query:
                    query_analysis["query_type"] = "comparison"
                    # IMPORTANT: Restore the original teams and drivers
                    if "teams" in clarification_context:
                        query_analysis["teams"] = clarification_context["teams"]
                    if "drivers" in clarification_context:
                        query_analysis["drivers"] = clarification_context["drivers"]
                elif "sector" in original_query:
                    query_analysis["query_type"] = "sector_analysis"
                elif "position" in original_query:
                    query_analysis["query_type"] = "position_analysis"
                elif "pit" in original_query:
                    query_analysis["query_type"] = "pit_strategy"
                elif "lap" in original_query or "time" in original_query:
                    query_analysis["query_type"] = "driver_performance"
                else:
                    query_analysis["query_type"] = "driver_performance"
                
                self.context_manager.clear_clarification_context()
                return None  # Continue with execution
                
            else:
                # Response was unclear, ask again using LLM
                follow_up_prompt = f"""
    The user's response "{user_query}" was unclear. Please ask them again in a friendly way 
    to clarify whether they want a visualization or text analysis. Be conversational and helpful.
    """
                
                try:
                    follow_up = self.llm.invoke(follow_up_prompt).content
                    return follow_up
                except Exception as e:
                    self.logger.warning(f"Failed to generate follow-up with LLM: {e}")
                    return "I didn't quite understand. Would you like a visualization chart or a text summary?"
                    
        except Exception as e:
            self.logger.warning(f"Failed to interpret clarification response with LLM: {e}")
            # Fallback to keyword matching
            viz_keywords = ["visualization", "chart", "graph", "plot", "visualize", "show me a"]
            text_keywords = ["summary", "analysis", "text", "data", "stats", "performance"]
            
            if any(keyword in query_lower for keyword in viz_keywords):
                query_analysis.update(clarification_context)
                query_analysis["query_type"] = "visualization"
                
                # Determine visualization type based on original query
                original_query = clarification_context.get("original_query", "").lower()
                if "sector" in original_query:
                    query_analysis["visualization_type"] = "sector_analysis"
                elif "position" in original_query:
                    query_analysis["visualization_type"] = "position_progression"
                elif "pit" in original_query:
                    query_analysis["visualization_type"] = "pit_stop_analysis"
                elif "lap" in original_query or "time" in original_query:
                    query_analysis["visualization_type"] = "lap_time_progression"
                else:
                    query_analysis["visualization_type"] = "lap_time_progression"
                
                self.context_manager.clear_clarification_context()
                return None
            elif any(keyword in query_lower for keyword in text_keywords):
                query_analysis.update(clarification_context)
                
                # Check if the original query was a comparison
                original_query = clarification_context.get("original_query", "").lower()
                if "compare" in original_query or "vs" in original_query or "versus" in original_query:
                    query_analysis["query_type"] = "comparison"
                    # IMPORTANT: Restore the original teams and drivers
                    if "teams" in clarification_context:
                        query_analysis["teams"] = clarification_context["teams"]
                    if "drivers" in clarification_context:
                        query_analysis["drivers"] = clarification_context["drivers"]
                else:
                    query_analysis["query_type"] = "driver_performance"
                
                self.context_manager.clear_clarification_context()
                return None
            else:
                return "I didn't quite understand. Would you like a visualization chart or a text summary?"

    def _execute_tools_for_query(self, user_query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tools based on query analysis with proper session detection"""
        tool_results = {}
        sql_queries = []
        
        try:
            meeting_info = query_analysis["meeting_info"]
            session_type = query_analysis["session_type"]
            query_type = query_analysis["query_type"]  # Add this line
            
            # Add debugging here
            self.logger.info(f"🔍 Executing tools for query type: {query_type}")
            self.logger.info(f"🔍 Meeting info: {meeting_info}")
            self.logger.info(f"🔍 Session type: {session_type}")
            
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
            drivers = query_analysis["drivers"]
            
            # Extract team names for all query types
            teams = query_analysis["teams"]
            
            # Add debugging for team extraction
            self.logger.info(f"�� Extracted teams: {teams}")
            self.logger.info(f"�� Extracted drivers: {drivers}")
            
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
            
            if query_type == "visualization":
                # Determine visualization type based on query content or stored context
                viz_type = query_analysis.get("visualization_type") or self._determine_visualization_type(user_query, query_analysis)
                
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
            
            elif query_type == "position_analysis":
                # Handle position analysis as text query
                tool_results["position_analysis"] = self.http_client.call_tool("get_position_progression", {
                    "session_key": session_key,
                    "driver_filter": driver_filter,
                    "team_filter": team_filter,
                    "progression_type": "all"
                })
                if "sql_query" in tool_results["position_analysis"]:
                    sql_queries.append({"tool": "get_position_progression", "query": tool_results["position_analysis"]["sql_query"], "params": tool_results["position_analysis"].get("sql_params", {})})
            
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
                    "driver_filter": driver_filter,  # Use the properly formatted driver_filter
                    "team_filter": team_filter,      # Use the properly formatted team_filter
                    "sector_analysis_type": sector_analysis_type
                })
                if "sql_query" in tool_results["sector_analysis"]:
                    sql_queries.append({"tool": "get_sector_analysis", "query": tool_results["sector_analysis"]["sql_query"], "params": tool_results["sector_analysis"].get("sql_params", {})})
            
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
                    return {"error": "Need to specify at least two drivers or teams for comparison"}
            
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

            if query_type == "team_performance":
                # Add debugging for team performance
                self.logger.info(f"🔍 Executing team performance for teams: {teams}")
                
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
                # Add driver_filter and team_filter to the fastest lap tool call
                tool_results["fastest_lap"] = self.http_client.call_tool("get_fastest_lap", {
                    "session_key": session_key,
                    "driver_filter": driver_filter,  # Add this line
                    "team_filter": team_filter       # Add this line
                })
                if "sql_query" in tool_results["fastest_lap"]:
                    sql_queries.append({"tool": "get_fastest_lap", "query": tool_results["fastest_lap"]["sql_query"], "params": tool_results["fastest_lap"].get("sql_params", {})})
            
            elif query_type == "tire_strategy":
                # Use the already-formatted driver_filter and team_filter from earlier in the method
                tool_results["tire_strategy"] = self.http_client.call_tool("get_tire_strategy", {
                    "session_key": session_key,
                    "driver_filter": driver_filter,  # ✅ Use the properly formatted string
                    "team_filter": team_filter,      # ✅ Use the properly formatted string
                    "strategy_type": "all"
                })
                if "sql_query" in tool_results["tire_strategy"]:
                    sql_queries.append({"tool": "get_tire_strategy", "query": tool_results["tire_strategy"]["sql_query"], "params": tool_results["tire_strategy"].get("sql_params", {})})

            elif query_type == "pit_strategy":
                # Use the already-formatted driver_filter and team_filter from earlier in the method
                tool_results["pit_stop_analysis"] = self.http_client.call_tool("get_pit_stop_analysis", {
                    "session_key": session_key,
                    "analysis_type": "all",
                    "driver_filter": driver_filter,  # ✅ Use the properly formatted string
                    "team_filter": team_filter       # ✅ Use the properly formatted string
                })
                
                if "sql_query" in tool_results["pit_stop_analysis"]:
                    sql_queries.append({
                        "tool": "get_pit_stop_analysis", 
                        "query": tool_results["pit_stop_analysis"]["sql_query"], 
                        "params": tool_results["pit_stop_analysis"].get("sql_params", {})
                    })
                        
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
                
                if "sector_analysis" not in tool_results:
                    tool_results["sector_analysis"] = self.http_client.call_tool("get_sector_analysis", {
                        "session_key": session_key,
                        "driver_filter": driver_filter,  # Use the properly formatted driver_filter
                        "team_filter": team_filter,      # Use the properly formatted team_filter
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
        
        self.logger.info(f"🔍 Extracting teams from: '{query_lower}'")
    
        for team_alias, team_name in team_mappings.items():
            if team_alias in query_lower:
                self.logger.info(f"✅ Found team alias '{team_alias}' -> '{team_name}'")
                if team_name not in found_teams:
                    found_teams.append(team_name)
        
        self.logger.info(f"🔍 Final extracted teams: {found_teams}")
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
    6. Do NOT mention filenames or technical details

    FORMATTING FOR VISUALIZATIONS:
    - Be friendly but concise
    - Do NOT list individual driver names unless specifically asked
    - Focus on what the visualization shows, not who is included
    - Use proper line breaks between sections
    - Keep the tone engaging but factual
    - Do NOT mention file paths, filenames, or technical implementation details

    Example format:
    I've created a lap time progression visualization for the Australian Grand Prix 2025 Race that shows how each driver's performance evolved throughout the session.

    The visualization below displays lap time progression for all drivers over the course of the race. You can see how lap times varied throughout the session, including pit stops and any incidents that affected performance.

    Graph analysis capabilities will be available soon to help you dive deeper into the performance data.

    Write your response in a friendly, conversational tone with proper line breaks and formatting.
    """

    def _get_text_query_prompt(self, user_query: str, query_analysis: Dict[str, Any], filtered_tool_results: Dict[str, Any], conversation_context: str) -> str:
        """Generate prompt specifically for text-based queries"""
        
        # Check if this is a comparison query
        if query_analysis.get("query_type") == "comparison" and "driver_comparison" in filtered_tool_results:
            return f"""{conversation_context}You are an expert F1 analyst. The user is asking for a comparison between specific drivers. Focus ONLY on comparing the two drivers mentioned.

    User Query: {user_query}
    Race: {query_analysis["meeting_info"]["name"]} {query_analysis["meeting_info"]["year"]}
    Session: {query_analysis["session_type"]}

    Driver Comparison Data:
    {json.dumps(filtered_tool_results.get("driver_comparison", {}), indent=2)}

    DRIVER COMPARISON RESPONSE REQUIREMENTS:
    1. Focus ONLY on comparing the two specific drivers that were requested
    2. Present their performance side by side
    3. Convert lap times from seconds to MM:SS.mmm format (e.g., 77.962 seconds becomes 01:17.962)
    4. Highlight who was faster, more consistent, etc.
    5. Do NOT mention other drivers or podium finishers unless specifically relevant
    6. Keep the response focused and concise

    FORMATTING FOR DRIVER COMPARISON:
    - Start with a brief introduction mentioning the two drivers being compared
    - Present each driver's key metrics (best lap, average lap, consistency)
    - Provide a direct comparison highlighting the differences
    - Keep it factual and focused on the requested comparison

    Example format:
    Max Verstappen vs Lewis Hamilton - Monaco GP 2025 Race Comparison:

    Max Verstappen (Red Bull Racing):
    - Best Lap: 01:14.230
    - Average Lap: 01:17.962
    - Consistency: 6.904 seconds

    Lewis Hamilton (Ferrari):
    - Best Lap: 01:14.089
    - Average Lap: 01:18.075
    - Consistency: 6.728 seconds

    Comparison: Lewis Hamilton was slightly faster with a best lap of 01:14.089 vs Verstappen's 01:14.230, but Verstappen had a better average lap time.

    Write your response focusing specifically on the requested driver comparison.
    """
        elif query_analysis.get("query_type") == "position_analysis" and "position_progression" in filtered_tool_results:
            return f"""{conversation_context}You are an expert F1 analyst. The user is asking for position progression analysis.

    User Query: {user_query}
    Race: {query_analysis["meeting_info"]["name"]} {query_analysis["meeting_info"]["year"]}
    Session: {query_analysis["session_type"]}
    Drivers: {query_analysis.get("drivers", [])}
    Position Progression Data:
    {json.dumps(filtered_tool_results.get("position_progression", {}), indent=2)}

    POSITION PROGRESSION ANALYSIS RESPONSE REQUIREMENTS:
    1. Analyze the position changes throughout the race for the specified driver(s)
    2. Identify key moments when positions changed significantly
    3. Explain the factors that led to position gains or losses
    4. Highlight strategic overtakes, pit stops, or incidents that affected positions
    5. Compare the driver's starting position to their finishing position
    6. Provide insights on race pace, consistency, and strategic decisions
    7. Mention any notable battles or defensive driving
    8. Analyze the overall race strategy and its effectiveness

    TIME FORMATTING REQUIREMENTS:
    - Convert timestamps to readable format: "Lap X" or "XX minutes into the race"
    - Use relative time references like "early in the race", "mid-race", "towards the end"
    - For specific lap references, use "Lap X" format
    - Avoid raw timestamps like "13:29:16.778"
    - Use natural language like "around the 15-minute mark" or "with 10 laps remaining"

    ANALYSIS STRUCTURE:
    - Start with an overview of the driver's race performance
    - Break down the race into key phases (start, middle, end)
    - Identify the most significant position changes and their causes
    - Compare performance to expectations and qualifying position
    - Provide insights on race strategy and execution
    - Conclude with an overall assessment of the driver's performance

    EXAMPLE TIME FORMATTING:
    Instead of: "He moved to 3rd place at 13:29:16.778"
    Use: "He moved to 3rd place early in the race" or "He moved to 3rd place around Lap 5"

    Instead of: "He dropped to 2nd place at 13:40:42.106"
    Use: "He dropped to 2nd place during his first pit stop" or "He dropped to 2nd place around the 20-minute mark"

    Write your response focusing specifically on the position progression analysis with properly formatted time references."""
        
        elif query_analysis.get("query_type") == "sector_analysis" and "sector_analysis" in filtered_tool_results:
            # Filter and summarize sector analysis data to reduce token count
            sector_data = filtered_tool_results.get("sector_analysis", {})
            if "sector_data" in sector_data:
                # Extract only the essential information for each driver
                summarized_data = []
                for driver_data in sector_data["sector_data"]:
                    if driver_data.get("driver") and driver_data.get("best_sector1"):
                        summarized_data.append({
                            "driver": driver_data["driver"],
                            "team": driver_data["team"],
                            "best_sector1": driver_data["best_sector1"],
                            "best_sector2": driver_data["best_sector2"],
                            "best_sector3": driver_data["best_sector3"],
                            "avg_sector1": driver_data.get("avg_sector1"),
                            "avg_sector2": driver_data.get("avg_sector2"),
                            "avg_sector3": driver_data.get("avg_sector3")
                        })
                
                sector_data["sector_data"] = summarized_data

            return f"""{conversation_context}You are an expert F1 analyst. The user is asking for a sector comparison between specific drivers. Focus ONLY on comparing the sector performance of the drivers mentioned.

    User Query: {user_query}
    Race: {query_analysis["meeting_info"]["name"]} {query_analysis["meeting_info"]["year"]}
    Session: {query_analysis["session_type"]}

    Sector Analysis Data:
    {json.dumps(sector_data, indent=2)}

    SECTOR COMPARISON RESPONSE REQUIREMENTS:
    1. Focus ONLY on comparing the sector performance of the specific drivers requested
    2. Present their sector times side by side for each sector (Sector 1, Sector 2, Sector 3)
    3. Convert sector times from seconds to MM:SS.mmm format (e.g., 28.701 seconds becomes 00:28.701)
    4. Highlight which driver was faster in each sector
    5. Identify each driver's strongest and weakest sectors
    6. Do NOT mention other drivers unless specifically relevant
    7. Keep the response focused and concise

    FORMATTING FOR SECTOR COMPARISON:
    - Start with a brief introduction mentioning the two drivers being compared
    - Present each driver's best sector times for all three sectors
    - Provide a direct comparison highlighting the differences in each sector
    - Identify which driver was faster in each sector
    - Keep it factual and focused on the requested sector comparison

    Example format:
    Charles Leclerc vs Lando Norris - British GP 2025 Sector Comparison:

    Charles Leclerc (Ferrari):
    - Best Sector 1: 00:28.701
    - Best Sector 2: 00:36.674
    - Best Sector 3: 00:24.312

    Lando Norris (McLaren):
    - Best Sector 1: 00:28.747
    - Best Sector 2: 00:36.673
    - Best Sector 3: 00:24.312

    Comparison: Charles Leclerc was faster in Sector 1 by 0.046 seconds, while Lando Norris was faster in Sector 2 by 0.001 seconds. Both drivers were equally fast in Sector 3.

    Write your response focusing specifically on the requested sector comparison.
    """
        elif query_analysis.get("query_type") == "comparison" and "team_comparison" in filtered_tool_results:
            return f"""{conversation_context}You are an expert F1 analyst. The user is asking for a comparison between specific teams. Focus ONLY on comparing the two teams mentioned.

    User Query: {user_query}
    Race: {query_analysis["meeting_info"]["name"]} {query_analysis["meeting_info"]["year"]}
    Session: {query_analysis["session_type"]}

    Team Comparison Data:
    {json.dumps(filtered_tool_results.get("team_comparison", {}), indent=2)}

    TEAM COMPARISON RESPONSE REQUIREMENTS:
    1. Focus ONLY on comparing the two specific teams that were requested
    2. Present their performance side by side
    3. Convert lap times from seconds to MM:SS.mmm format (e.g., 77.962 seconds becomes 01:17.962)
    4. Highlight which team was faster, more consistent, etc.
    5. Include individual driver performance within each team
    6. Do NOT mention other teams or podium finishers unless specifically relevant
    7. Keep the response focused and concise

    FORMATTING FOR TEAM COMPARISON:
    - Start with a brief introduction mentioning the two teams being compared
    - Present each team's overall performance (total laps, average lap, best lap, consistency)
    - Include individual driver performance within each team
    - Provide a direct comparison highlighting the differences
    - Keep it factual and focused on the requested team comparison

    Example format:
    Mercedes vs Red Bull Racing - Monaco GP 2025 Race Comparison:

    Mercedes:
    - Total Laps: 151
    - Average Lap: 01:20.194
    - Best Lap: 01:13.404
    - Consistency: 6.441 seconds
    - Best Position: 11th
    - Drivers: George Russell (best lap: 01:13.404) and Kimi Antonelli (best lap: 01:13.518)

    Red Bull Racing:
    - Total Laps: 154
    - Average Lap: 01:19.028
    - Best Lap: 01:14.230
    - Consistency: 6.404 seconds
    - Best Position: 4th
    - Drivers: Max Verstappen (best lap: 01:14.230) and Yuki Tsunoda (best lap: 01:14.913)

    Comparison: Red Bull Racing had better overall performance with a lower average lap time and better finishing positions, while Mercedes showed slightly better consistency.

    Write your response focusing specifically on the requested team comparison.
    """

        # Check if this is a driver performance query
        elif query_analysis.get("query_type") == "driver_performance" and "driver_performance" in filtered_tool_results:
            return f"""{conversation_context}You are an expert F1 analyst. The user is asking about a specific driver's performance. Focus ONLY on that driver's performance data.

    User Query: {user_query}
    Race: {query_analysis["meeting_info"]["name"]} {query_analysis["meeting_info"]["year"]}
    Session: {query_analysis["session_type"]}

    Driver Performance Data:
    {json.dumps(filtered_tool_results.get("driver_performance", {}), indent=2)}

    DRIVER PERFORMANCE RESPONSE REQUIREMENTS:
    1. Focus ONLY on the specific driver's performance that was requested
    2. Include their total laps, average lap time, best lap time, and consistency
    3. Convert lap times from seconds to MM:SS.mmm format (e.g., 77.962 seconds becomes 01:17.962)
    4. Mention their team and any notable aspects of their performance
    5. Do NOT mention other drivers unless specifically relevant to the requested driver's performance
    6. Keep the response focused and concise

    FORMATTING FOR DRIVER PERFORMANCE:
    - Start with the driver's name and team
    - Include their total laps completed
    - Show their best lap time and average lap time in MM:SS.mmm format
    - Mention consistency if notable
    - Keep it factual and focused on the requested driver

    Example format:
    Max Verstappen (Red Bull Racing) completed 78 laps in the Monaco Grand Prix 2025 Race. His best lap was 01:14.230 and his average lap time was 01:17.962. He showed good consistency with a standard deviation of 6.904 seconds across all his laps.

    Write your response focusing specifically on the requested driver's performance.
    """
        elif query_analysis.get("query_type") == "tire_strategy" and "tire_strategy" in filtered_tool_results:
            return f"""{conversation_context}You are an expert F1 analyst. The user is asking about tire strategy analysis. Focus on the tire strategy data and stint information.

    User Query: {user_query}
    Race: {query_analysis["meeting_info"]["name"]} {query_analysis["meeting_info"]["year"]}
    Session: {query_analysis["session_type"]}

    Tire Strategy Data:
    {json.dumps(filtered_tool_results.get("tire_strategy", {}), indent=2)}

    TIRE STRATEGY RESPONSE REQUIREMENTS:
    1. Focus on tire compound usage, stint lengths, and strategy decisions
    2. Include information about tire compounds used (Soft, Medium, Hard)
    3. Mention stint durations and any notable tire management
    4. If specific teams/drivers were requested, focus on their tire strategy
    5. Include any pit stop timing related to tire changes
    6. Keep the response focused and concise

    FORMATTING FOR TIRE STRATEGY:
    - Start with an overview of tire strategy for the session
    - Include information about compounds used and stint lengths
    - If comparing teams/drivers, show their tire strategy side by side
    - Mention any notable tire management or strategy decisions
    - Keep it factual and focused on tire strategy

    Example format:
    For the Monaco Grand Prix 2025 Race, the tire strategy involved primarily Medium and Hard compounds. Most drivers opted for a two-stop strategy, with the first stint averaging 25 laps on Medium tires.

    Write your response focusing specifically on tire strategy and compound usage.
    """
        
        elif query_analysis.get("query_type") == "pit_strategy" and "pit_stop_analysis" in filtered_tool_results:
            return f"""{conversation_context}You are an expert F1 analyst. The user is asking about pit stop strategy analysis. Focus on the pit stop data and strategy insights.

        User Query: {user_query}
        Race: {query_analysis["meeting_info"]["name"]} {query_analysis["meeting_info"]["year"]}
        Session: {query_analysis["session_type"]}

        Pit Stop Analysis Data:
        {json.dumps(filtered_tool_results.get("pit_stop_analysis", {}), indent=2)}

        PIT STRATEGY RESPONSE REQUIREMENTS:
        1. Focus on pit stop timing, frequency, and strategy
        2. Include total pit stops, average pit stop duration, and fastest pit stops
        3. Convert times from seconds to MM:SS.mmm format (e.g., 2.5 seconds becomes 00:02.500)
        4. Mention any notable pit stop strategies or timing decisions
        5. If specific teams/drivers were requested, focus on their pit stop performance
        6. Keep the response focused and concise

        FORMATTING FOR PIT STRATEGY:
        - Start with an overview of pit stop activity
        - Include total number of pit stops and average duration
        - Highlight fastest pit stops and any notable strategies
        - If comparing teams/drivers, show their pit stop performance side by side
        - Keep it factual and focused on pit stop strategy

        Example format:
        For the Monaco Grand Prix 2025 Race, there were 15 pit stops with an average duration of 00:02.847 seconds. The fastest pit stop was completed by Red Bull Racing in 00:02.123 seconds.

        Write your response focusing specifically on pit stop strategy and timing.
        """
        # Default prompt for other query types
        else:
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

    Write your response in a direct, professional tone with proper line breaks and formatting.
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