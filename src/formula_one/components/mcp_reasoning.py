import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import time
import os
import asyncio
import inspect
import concurrent.futures

from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.config_entity import DatabaseConfig
from src.formula_one.entity.mcp_config_entity import MCPConfig
from src.formula_one.utils.query_validator import QueryValidator, ValidationResult

from src.formula_one.components.enhanced_context_memory import EnhancedContextManager
from src.formula_one.components.chart_analysis import ChartAnalyzer


class IntentAnalyzer(BaseComponent):
    """Analyze user query intent and extract information"""
    
    def __init__(self, config, db_config: DatabaseConfig):
        super().__init__(config, db_config)
        
        # Initialize LLM for intent classification
        api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY')
        self.intent_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,  # Lower temperature for more consistent classification
            openai_api_key=api_key,
            max_tokens=500
        )
        
        # Define intent patterns for keyword matching
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
            ],
            "chart_analysis": [
                "analyze this chart", "what does this chart show", "explain this visualization",
                "what can you tell me about this chart", "interpret this graph",
                "what insights can you provide", "analyze the data", "chart analysis",
                "what patterns do you see", "explain the trends", "what's happening in this chart"
            ]
        }

    def analyze_query_intent(self, user_query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Analyze user query to determine intent and extract relevant information"""
        
        # Add debugging
        self.logger.info(f"🔍 Analyzing query intent for: '{user_query}'")
        
        query_lower = user_query.lower()
        
        # Extract meeting information
        meeting_info = self._extract_meeting_info(user_query)
        
        # Determine session type
        session_type = self._determine_session_type(query_lower)
        
        # Use LLM for enhanced intent classification
        llm_intent = self._classify_intent_with_llm(user_query, conversation_history)
        
        # Fallback to rule-based classification if LLM fails
        if not llm_intent:
            llm_intent = self._determine_query_type(query_lower)
        
        # Add debug logging for pit stop analysis
        if "pit stop" in user_query.lower():
            self.logger.info(f"🔍 Pit stop query detected - LLM intent: {llm_intent}")
            self.logger.info(f"🔍 Rule-based intent would be: {self._determine_query_type(user_query.lower())}")
        
        # Extract entities
        drivers = self._extract_driver_names(user_query)
        teams = self._extract_team_names(user_query)
        lap_number = self._extract_lap_number(user_query)
        
        # Get context from history
        context = self._get_context_from_history(conversation_history) if conversation_history else {}
        
        return {
            "query_type": llm_intent,
            "meeting_info": meeting_info,
            "session_type": session_type,
            "drivers": drivers,
            "teams": teams,
            "lap_number": lap_number,
            "context": context,
            "original_query": user_query
        }

    def _classify_intent_with_llm(self, user_query: str, conversation_history: List[Dict] = None) -> str:
        """Use LLM to classify user intent more accurately"""
        
        # Skip LLM classification for very short queries to avoid meta-commentary
        if len(user_query.strip()) <= 3:
            self.logger.info(f"🔍 Skipping LLM classification for short query: '{user_query}'")
            return None
        
        # Build context from conversation history
        context_prompt = ""
        if conversation_history:
            recent_context = conversation_history[-3:]  # Last 3 exchanges
            context_prompt = f"\nRecent conversation context:\n"
            for exchange in recent_context:
                context_prompt += f"User: {exchange.get('user_message', exchange.get('query', ''))}\n"
                response_text = exchange.get('bot_response', exchange.get('response', ''))
                context_prompt += f"Assistant: {response_text[:100]}...\n"
        
        classification_prompt = f"""You are an expert F1 data analyst. Classify the user's intent from the following query.

Available intent types:
- comparison: Comparing drivers, teams, or performance metrics
- driver_performance: Individual driver analysis and statistics
- team_performance: Team-level analysis and statistics
- fastest_lap: Best lap times and fastest laps
- sector_analysis: Sector-by-sector performance analysis
- tire_strategy: Tire compound usage and strategy
- pit_strategy: Pit stop timing and strategy
- incident_investigation: Accidents, incidents, or unusual events
- qualifying_results: Qualifying session results and grid positions
- race_results: Race outcomes and finishing positions
- visualization: Requests for charts, graphs, or visual data
- position_analysis: Position progression and race position analysis
- lap_time_progression: Lap time trends and progression

{context_prompt}

User Query: "{user_query}"

Instructions:
1. Analyze the user's intent carefully
2. Consider context from recent conversation
3. Return ONLY the intent type (e.g., "comparison", "driver_performance", etc.)
4. If unclear, choose the most likely intent based on keywords and context

Intent:"""

        try:
            response = self.intent_llm.invoke(classification_prompt)
            intent = response.content.strip().lower()
            
            # Validate the intent
            valid_intents = list(self.intent_patterns.keys()) + ["position_analysis", "lap_time_progression"]
            if intent in valid_intents:
                self.logger.info(f"🔍 LLM classified intent as: {intent}")
                return intent
            else:
                self.logger.warning(f"🔍 LLM returned invalid intent: {intent}, falling back to rule-based")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in LLM intent classification: {e}")
            return None


    
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
        """Extract meeting name and year from query with intelligent abbreviation handling"""
        
        # Base race mappings (without abbreviations)
        race_mappings = {
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
            'belgian': 'Belgian Grand Prix',
            'spa': 'Belgian Grand Prix',
            'hungarian': 'Hungarian Grand Prix',
            'hungary': 'Hungarian Grand Prix',
            'hungarian grand prix': 'Hungarian Grand Prix'
        }
        
        query_lower = query.lower()
        meeting_name = None
        
        # First, try exact matches from the base mappings
        for keyword, race_name in race_mappings.items():
            if keyword in query_lower:
                meeting_name = race_name
                break
        
        # If no exact match found, try intelligent abbreviation handling
        if not meeting_name:
            # Look for patterns like "monaco gp", "austrian gp", etc.
            
            # Pattern to match: word + "gp" or "grand prix"
            gp_patterns = [
                r'(\w+)\s+(?:gp|grand\s+prix)',
                r'(?:gp|grand\s+prix)\s+(\w+)'
            ]
            
            for pattern in gp_patterns:
                matches = re.findall(pattern, query_lower)
                for match in matches:
                    # Try to map the base word to a race
                    base_word = match.strip()
                    for keyword, race_name in race_mappings.items():
                        if base_word in keyword or keyword in base_word:
                            meeting_name = race_name
                            break
                    if meeting_name:
                        break
                if meeting_name:
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
        # Add null check to prevent TypeError
        if not conversation_history:
            return {"name": None, "year": 2025}
        
        for entry in reversed(conversation_history[-3:]):
            previous_meeting = self._extract_meeting_info(entry.get("user_message", entry.get("message", "")))
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
            "Mercedes": ["mercedes", "mercedes team", "mercedes amg"],
            "Ferrari": ["ferrari", "ferrari team", "scuderia ferrari"],
            "Red Bull Racing": ["red bull", "red bull racing", "red bull team", "redbull"],
            "McLaren": ["mclaren", "mclaren team", "mclaren racing"],
            "Aston Martin": ["aston martin", "aston martin team", "aston"],
            "Alpine": ["alpine", "alpine team", "alpine racing"],
            "Williams": ["williams", "williams team", "williams racing"],
            "AlphaTauri": ["alphatauri", "alpha tauri", "alpha tauri team"],
            "Alfa Romeo": ["alfa romeo", "alfa", "alfa romeo team"],
            "Haas": ["haas", "haas team", "haas f1"],
            "Sauber": ["sauber", "sauber team", "sauber racing"],
            "Racing Bulls": ["racing bulls", "racing bulls team", "bulls"]
        }
        
        query_lower = query.lower()
        found_teams = []
        
        # First, try to find valid teams
        for full_name, aliases in team_mappings.items():
            for alias in aliases:
                if alias in query_lower:
                    if full_name not in found_teams:
                        found_teams.append(full_name)
                    break
        
        # If no valid teams found, look for potential invalid team names
        if not found_teams:
            # Look for patterns that suggest a team name was mentioned
            import re
            
            # First, get valid driver names to avoid false positives
            valid_driver_names = [
                "max verstappen", "lewis hamilton", "charles leclerc", "lando norris",
                "oscar piastri", "george russell", "fernando alonso", "carlos sainz",
                "pierre gasly", "yuki tsunoda", "alex albon", "lance stroll",
                "esteban ocon", "nico hulkenberg", "liam lawson", "oliver bearman",
                "gabriel bortoleto", "isack hadjar", "jack doohan", "franco colapinto",
                "andrea kimi antonelli"
            ]
            
            # More specific patterns that clearly indicate a team name
            team_patterns = [
                r'team\s+([A-Za-z]+\s+[A-Za-z]+)',      # "team Red Bull"
                r'team\s+([A-Za-z]+)',                   # "team Red"
            ]
            
            for pattern in team_patterns:
                matches = re.findall(pattern, query_lower)
                for match in matches:
                    # Check if this looks like a team name but isn't in our valid list
                    potential_team = match.strip().lower()
                    if potential_team and len(potential_team) > 2:  # At least 3 characters
                        # Check if it's not a common word that might be part of the query
                        common_words = ['the', 'and', 'for', 'at', 'in', 'on', 'with', 'from', 'to', 'of', 'a', 'an', 'time', 'lap', 'progression', 'analysis', 'performance', 'strategy', 'pit', 'stop', 'tire', 'tyre', 'sector', 'position', 'qualifying', 'practice', 'race', 'sprint', 'grand', 'prix', 'gp']
                        if potential_team not in common_words and potential_team not in valid_driver_names:
                            # Only flag if it looks like a proper team name (contains uppercase or is clearly a name)
                            if any(word[0].isupper() for word in match.split()) or len(match.split()) >= 2:
                                # This looks like an invalid team name
                                found_teams.append(match.title())
                                break
        
        return found_teams

    def _determine_query_type(self, query_lower: str) -> str:
        """Determine query type using semantic similarity with priority logic"""
        
        # Add debugging
        self.logger.info(f"🔍 Determining query type for: '{query_lower}'")
        
        # First, check for chart analysis keywords (highest priority)
        chart_analysis_keywords = [
            "analyze this chart", "what does this chart show", "explain this visualization",
            "what can you tell me about this chart", "interpret this graph",
            "what insights can you provide", "analyze the data", "chart analysis",
            "what patterns do you see", "explain the trends", "what's happening in this chart"
        ]
        
        for keyword in chart_analysis_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched chart analysis keyword: '{keyword}'")
                return "chart_analysis"
        
        # Check for visualization keywords (high priority)
        viz_keywords = [
            "show me a graph", "create a chart", "visualize", "plot", "graph",
            "show me the data", "display", "chart", "visualization", "graphical",
            "show me a visualization", "create a plot", "show the trend",
            "plot the data", "graph the results", "show me the progression",
            "visualize the data", "create a diagram", "show me the analysis",
            "show me", "sector chart", "position chart", "lap chart", "tire chart",
            "pit stop chart", "strategy chart", "create a", "show me a"
        ]
        
        for keyword in viz_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched visualization keyword: '{keyword}'")
                return "visualization"
        
        # Check for lap time progression keywords
        lap_time_keywords = [
            "lap time progression", "lap times", "lap time", "lap analysis",
            "lap comparison", "lap performance", "lap time chart", "lap time analysis",
            "lap time tracking", "lap time development", "lap time evolution"
        ]
        for keyword in lap_time_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched lap time progression keyword: '{keyword}'")
                return "lap_time_progression"
        
        position_progression_keywords = [
            "position progression", "position changes", "position analysis",
            "how did positions change", "position tracking", "position movement",
            "position development", "position evolution", "position history",
            "position chart", "position analysis"
        ]
        for keyword in position_progression_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched position progression keyword: '{keyword}'")
                return "position_analysis"
            
        # Check for tire strategy keywords (HIGHER PRIORITY than comparison)
        tire_strategy_keywords = [
            "tire strategy", "tire strategies", "tire compound", "tire compounds",
            "tire usage", "tire management", "tire performance", "tire analysis",
            "compound strategy", "compound usage", "compound management",
            "stint", "stints", "stint length", "stint duration",
            "tire comparison", "tire strategy comparison", "compound comparison",
            "tire compounds used", "tire strategy for"
        ]
        for keyword in tire_strategy_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched tire strategy keyword: '{keyword}'")
                return "tire_strategy"
        
        # Check for pit stop keywords
        pit_stop_keywords = [
            "pit stop", "pitstop", "pitted", "when did they pit", "pit strategy",
            "pit stops", "pitstop", "when did", "pit timing", "pit duration",
            "pit stop analysis", "pit stop strategy", "pit stop timing",
            "when drivers pitted", "pit stop times", "pit stop performance",
            "pit stop comparison", "pit stop for"
        ]
        for keyword in pit_stop_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched pit stop keyword: '{keyword}'")
                return "pit_strategy"
        
        # Check for fastest lap keywords
        fastest_lap_keywords = [
            "fastest lap", "best lap", "quickest lap", "fastest lap time",
            "best lap time", "quickest lap time", "fastest lap for",
            "what was the fastest", "fastest lap at"
        ]
        for keyword in fastest_lap_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched fastest lap keyword: '{keyword}'")
                return "fastest_lap"
        
        # Check for qualifying results keywords
        qualifying_keywords = [
            "qualifying results", "qualifying positions", "qualifying outcome",
            "qualifying for", "qualifying at", "qualifying performance"
        ]
        for keyword in qualifying_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched qualifying keyword: '{keyword}'")
                return "qualifying_results"
        
        # Check for race results keywords
        race_results_keywords = [
            "race results", "final positions", "race outcome", "race for",
            "race at", "final results", "race positions", "race finish"
        ]
        for keyword in race_results_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched race results keyword: '{keyword}'")
                return "race_results"
        
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
        driver_keywords = ["driver performance", "driver analysis", "driver stats", "driver data", "how did", "performance for"]
        for keyword in driver_keywords:
            if keyword in query_lower:
                self.logger.info(f"✅ Matched driver performance keyword: '{keyword}'")
                return "driver_performance"
        
        # If no explicit keywords matched, default to race_results
        self.logger.info(f"🔍 No explicit keywords matched, defaulting to race_results")
        return "race_results"

    
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
        
        # First, try to find valid drivers
        for full_name, aliases in driver_mappings.items():
            for alias in aliases:
                if alias in query_lower:
                    if full_name not in found_drivers:
                        found_drivers.append(full_name)
                    break
        
        # If no valid drivers found, look for potential invalid driver names
        if not found_drivers:
            # Look for patterns that suggest a driver name was mentioned
            import re
            # More specific patterns that clearly indicate a driver name
            driver_patterns = [
                r'for\s+([A-Za-z]+\s+[A-Za-z]+)\s+at',  # "for Max Verstappen at"
                r'driver\s+([A-Za-z]+\s+[A-Za-z]+)',    # "driver Max Verstappen"
                r'driver\s+([A-Za-z]+)',                 # "driver Max"
            ]
            
            for pattern in driver_patterns:
                matches = re.findall(pattern, query_lower)
                for match in matches:
                    # Check if this looks like a driver name but isn't in our valid list
                    potential_driver = match.strip()
                    if potential_driver and len(potential_driver) > 2:  # At least 3 characters
                        # Check if it's not a common word that might be part of the query
                        common_words = ['the', 'and', 'for', 'at', 'in', 'on', 'with', 'from', 'to', 'of', 'a', 'an', 'time', 'lap', 'progression', 'analysis', 'performance', 'strategy', 'pit', 'stop', 'tire', 'tyre', 'sector', 'position', 'qualifying', 'practice', 'race', 'sprint', 'grand', 'prix', 'gp']
                        if potential_driver not in common_words:
                            # Only flag if it looks like a proper name (contains uppercase or is clearly a name)
                            if any(word[0].isupper() for word in potential_driver.split()) or len(potential_driver.split()) >= 2:
                                # This looks like an invalid driver name
                                found_drivers.append(potential_driver.title())
                                break
        
        return found_drivers 

class ReasoningEngine(BaseComponent):
    """Main reasoning engine for processing F1 queries"""
    
    def __init__(self, config, db_config: DatabaseConfig, tools: Dict[str, Any], http_client):
        super().__init__(config, db_config)
        self.tools = tools
        self.http_client = http_client
        self.intent_analyzer = IntentAnalyzer(config, db_config)
        self.context_manager = EnhancedContextManager(config, db_config)  # Use enhanced context manager
        
        # Initialize chart analyzer
        self.chart_analyzer = ChartAnalyzer(config, db_config)

        self.logger.info(f"🔍 Config openai_api_key type: {type(config.openai_api_key)}")
        self.logger.info(f"🔍 Config openai_api_key value: {config.openai_api_key}")
        
        # Try to get from environment directly as fallback
        env_key = os.getenv('OPENAI_API_KEY')
        if not config.openai_api_key and env_key:
            self.logger.info("🔧 Using environment variable as fallback")
            api_key = env_key
        else:
            api_key = config.openai_api_key
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=api_key
        )
        
        # Add reasoning configuration
        self.reasoning_config = {
            "enable_chain_of_thought": True,
            "enable_few_shot_examples": True,
            "enable_context_awareness": True,
            "max_reasoning_steps": 5,
            "confidence_threshold": 0.7
        }
        
        # Initialize query validator
        self.query_validator = QueryValidator(db_config)

    def reason_and_answer(self, user_query: str) -> tuple:
        """Main reasoning method that analyzes query and generates response"""
        try:
            self.logger.info(f"🧠 Processing query: '{user_query}'")
            
            # Initialize query_analysis variable
            query_analysis = {}
            
            # Get conversation history from context manager
            conversation_history = self.context_manager.conversation_history
            
            # Add debug logging
            self.logger.info(f"🔍 Conversation history length: {len(conversation_history)}")
            if conversation_history:
                self.logger.info(f"🔍 Last conversation entry: {conversation_history[-1]}")
                
                # Check if the last response was asking for clarification
                last_response = conversation_history[-1].get("bot_response", conversation_history[-1].get("response", ""))
                self.logger.info(f"🔍 Last response contains clarification patterns: {any(pattern in last_response for pattern in ['Could you please specify', 'I cannot identify'])}")
            
            # Check if this is a clarification response to a previous query
            # Only treat as clarification if the query is not too short and actually looks like a clarification
            if conversation_history and len(user_query.strip()) > 3 and self.context_manager.is_clarification_response(user_query):
                self.logger.info(f"🔍 Detected clarification response: {user_query}")
                self.logger.info(f"🔍 Conversation history length: {len(conversation_history)}")
                if conversation_history:
                    last_response_text = conversation_history[-1].get('bot_response', conversation_history[-1].get('response', ''))
                self.logger.info(f"🔍 Last response: {last_response_text[:200]}")
                
                # Check what type of clarification this is
                last_response = conversation_history[-1].get("bot_response", conversation_history[-1].get("response", ""))
                clarification_context = self.context_manager.get_clarification_context()
                
                # Check if this is a visualization vs summary clarification response FIRST
                viz_clarification_patterns = [
                    "Would you like me to create a visualization",
                    "Would you like me to create a chart",
                    "visualization chart",
                    "text summary"
                ]
                
                is_viz_clarification = any(pattern in last_response for pattern in viz_clarification_patterns)
                
                if is_viz_clarification and clarification_context:
                    self.logger.info(f"🔍 Visualization clarification handled, continuing with execution")
                    # Create a query_analysis object to be updated by the clarification handler
                    query_analysis = {}
                    self.logger.info(f"🔍 Calling _handle_clarification_response with user_query: '{user_query}'")
                    clarification_result = self._handle_clarification_response(user_query, query_analysis)
                    self.logger.info(f"🔍 _handle_clarification_response returned: {clarification_result}")
                    self.logger.info(f"🔍 Updated query_analysis: {query_analysis}")
                    if clarification_result:
                        # Still need more clarification
                        self.context_manager.add_to_history(user_query, clarification_result, {})
                        return clarification_result, None
                    else:
                        # Clarification handled, continue with execution
                        self.logger.info(f"🔍 Visualization clarification handled, continuing with execution")
                        # The query_analysis has been updated by _handle_clarification_response
                        # Continue with the normal flow using the updated query_analysis
                        # Skip the normal flow since we have the updated query_analysis
                        return self._continue_with_updated_analysis(user_query, query_analysis, conversation_history)
                        
                # Check if this is a race clarification response
                race_clarification_patterns = [
                    "I couldn't identify which race",
                    "Could you please specify the race",
                    "Could you please specify which race",
                    "which race you're interested in",
                    "which race you would like",
                    "which specific race",
                    "which race you're referring to",
                    "specify which race",
                    "specify the race",
                    "please specify",
                    "could you please specify",
                    "which race",
                    "specific race"
                ]
                
                is_race_clarification = any(pattern in last_response for pattern in race_clarification_patterns)
                self.logger.info(f"🔍 Race clarification check - last response: '{last_response[:100]}...'")
                self.logger.info(f"🔍 Is race clarification: {is_race_clarification}")
                
                # Debug: show which pattern matched
                if is_race_clarification:
                    for pattern in race_clarification_patterns:
                        if pattern in last_response:
                            self.logger.info(f"🔍 Matched race clarification pattern: '{pattern}'")
                            break
                
                if is_race_clarification:
                    self.logger.info(f"🔍 Handling race clarification response")
                    # Extract race information from the user's response
                    self.logger.info(f"🔍 Extracting race info from: '{user_query}'")
                    extracted_meeting_info = self.intent_analyzer._extract_meeting_info(user_query)
                    self.logger.info(f"🔍 Extracted meeting info: {extracted_meeting_info}")
                    
                    if extracted_meeting_info.get("name"):
                        self.logger.info(f"🔍 Extracted race from clarification: {extracted_meeting_info}")
                        
                        # Get the original query context
                        original_query = clarification_context.get("original_query", "")
                        
                        # Create new query analysis with the extracted race info
                        # Re-analyze the original query to determine the specific visualization type
                        original_query = clarification_context.get("original_query", "")
                        reanalyzed_query = self.intent_analyzer.analyze_query_intent(original_query)
                        
                        # Update the query analysis with the extracted race info and reanalyzed query type
                        query_analysis = {
                            "query_type": reanalyzed_query.get("query_type", clarification_context.get("query_type", "")),
                            "meeting_info": extracted_meeting_info,
                            "session_type": clarification_context.get("session_type", "Race"),
                            "drivers": clarification_context.get("drivers", []),
                            "teams": clarification_context.get("teams", []),
                            "lap_number": None,
                            "context": {},
                            "confidence": 0.9,
                            "original_query": original_query
                        }
                        
                        # Clear the clarification context
                        self.context_manager.clear_clarification_context()
                        
                        # Continue with tool execution using the updated context
                        self.logger.info(f"🔍 Using extracted race info for clarification: {query_analysis}")
                        
                        # Execute tools with the updated context
                        tool_results = self._execute_tools_with_reasoning(original_query, query_analysis)
                        
                        # Generate response
                        response = self._generate_intelligent_summary_with_reasoning(
                            original_query, query_analysis, tool_results
                        )
                        
                        # Extract visualization data if present
                        visualization_data = None
                        visualization_tools = [
                            "create_lap_time_progression", "create_position_progression", 
                            "create_sector_analysis", "create_pit_stop_analysis", "create_tire_strategy"
                        ]
                        
                        for tool_name in visualization_tools:
                            if tool_name in tool_results and tool_results[tool_name].get("success"):
                                visualization_data = tool_results[tool_name]
                                self.logger.info(f"🔍 Found visualization data: {visualization_data}")
                                break
                        
                        # Store the exchange
                        self.context_manager.add_to_history(original_query, response, query_analysis)
                        
                        return response, visualization_data
                    else:
                        # Couldn't extract race info, ask again
                        return "I couldn't identify the race from your response. Please specify the race name (e.g., 'Monaco GP', 'Bahrain GP', etc.)", None
                
                # Check if this is a visualization vs summary clarification response
                viz_clarification_patterns = [
                    "Would you like me to create a visualization",
                    "Would you like me to create a chart",
                    "visualization chart",
                    "text summary"
                ]
                
                is_viz_clarification = any(pattern in last_response for pattern in viz_clarification_patterns)
                
                if is_viz_clarification and clarification_context:
                    self.logger.info(f"🔍 Visualization clarification handled, continuing with execution")
                    # Create a query_analysis object to be updated by the clarification handler
                    query_analysis = {}
                    clarification_result = self._handle_clarification_response(user_query, query_analysis)
                    if clarification_result:
                        # Still need more clarification
                        self.context_manager.add_to_history(user_query, clarification_result, {})
                        return clarification_result, None
                    else:
                        # Clarification handled, continue with execution
                        self.logger.info(f"🔍 Visualization clarification handled, continuing with execution")
                        # The query_analysis has been updated by _handle_clarification_response
                        # Continue with the normal flow using the updated query_analysis
                        
            # Analyze the query with conversation history (only if not handled by clarification)
            if not query_analysis or not query_analysis.get("query_type"):  # Only analyze if not already handled by clarification
                query_analysis = self.intent_analyzer.analyze_query_intent(user_query, conversation_history)
            self.logger.info(f"🧠 Query analysis: {query_analysis}")
            
            # Validate the query and query analysis
            try:
                validation_result = self.query_validator.validate_query(user_query, query_analysis)
                if not validation_result.is_valid:
                    error_response = self.query_validator.format_error_response(validation_result, query_analysis.get("query_type"))
                    self.logger.warning(f"❌ Query validation failed: {validation_result.errors}")
                    # Add the error response to conversation history
                    self.context_manager.add_to_history(user_query, error_response, query_analysis)
                    return error_response, None
                
                # Log warnings if any
                if validation_result.warnings:
                    self.logger.warning(f"⚠️ Query validation warnings: {validation_result.warnings}")
            except Exception as e:
                self.logger.error(f"❌ Validation error: {e}")
                # If validation fails, generate a basic error response for short queries
                if len(user_query.strip()) <= 3:
                    error_response = "Your query is too short. Please provide a complete query like 'lap time progression for Max Verstappen at Monaco GP'"
                    self.context_manager.add_to_history(user_query, error_response, query_analysis)
                    return error_response, None
                else:
                    # For other validation errors, continue with the query anyway
                    self.logger.info("🔍 Continuing with query despite validation error")
            
            # Get enhanced context from ChromaDB
            enhanced_context = self.context_manager.get_enhanced_context(user_query, query_analysis)
            self.logger.info(f"🔍 Enhanced context: {enhanced_context}")
            
            # Check if meeting is identified
            meeting_info = query_analysis.get("meeting_info", {})
            self.logger.info(f"🔍 Meeting info check: {meeting_info}")
            if not meeting_info or not meeting_info.get("name"):
                self.logger.info(f"🔍 No meeting identified, generating clarification")
                
                # Store context for when user provides clarification
                self.context_manager.set_clarification_context({
                    "query_type": query_analysis.get("query_type"),
                    "meeting_info": meeting_info,
                    "session_type": query_analysis.get("session_type"),
                    "drivers": query_analysis.get("drivers", []),
                    "teams": query_analysis.get("teams", []),
                    "original_query": user_query
                })
                
                self.logger.info(f"🔍 Stored clarification context: {self.context_manager.get_clarification_context()}")
                
                # Generate a direct, helpful clarification response
                response = f"I couldn't identify which race you're asking about. Could you please specify the race? For example: 'Bahrain Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix', etc."
                self.context_manager.add_to_history(user_query, response, query_analysis)
                self.logger.info(f"🔍 Added clarification request to history")
                return response, None
            
            # Check for ambiguous queries that need clarification about visualization vs summary
            clarification_needed = self._check_for_ambiguous_query(user_query, query_analysis)
            if clarification_needed:
                self.logger.info(f"🔍 Ambiguous query detected, asking for clarification")
                self.context_manager.add_to_history(user_query, clarification_needed, query_analysis)
                return clarification_needed, None
            
            # Execute tools with enhanced reasoning
            tool_results = self._execute_tools_with_reasoning(user_query, query_analysis)
            
            # Generate intelligent response
            response = self._generate_intelligent_summary_with_reasoning(user_query, query_analysis, tool_results)
            
            # Add to conversation history with enhanced context
            self.context_manager.add_to_history(user_query, response, query_analysis)
            
            # Extract visualization data if present
            visualization_data = None
            visualization_tools = [
                "create_lap_time_progression", "create_position_progression", 
                "create_sector_analysis", "create_pit_stop_analysis", "create_tire_strategy"
            ]
            
            for tool_name in visualization_tools:
                if tool_name in tool_results and tool_results[tool_name].get("success"):
                    visualization_data = tool_results[tool_name]
                    self.logger.info(f"🔍 Found visualization data: {visualization_data}")
                    break
            
            return response, visualization_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in reasoning: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}", None

    def _continue_with_updated_analysis(self, user_query: str, query_analysis: Dict[str, Any], conversation_history: List[Dict]) -> tuple:
        """Continue processing with updated query analysis from clarification handling"""
        try:
            self.logger.info(f"🧠 Continuing with updated query analysis: {query_analysis}")
            
            # Validate the updated query analysis
            try:
                validation_result = self.query_validator.validate_query(user_query, query_analysis)
                if not validation_result.is_valid:
                    error_response = self.query_validator.format_error_response(validation_result, query_analysis.get("query_type"))
                    self.logger.warning(f"❌ Query validation failed in updated analysis: {validation_result.errors}")
                    # Add the error response to conversation history
                    self.context_manager.add_to_history(user_query, error_response, query_analysis)
                    return error_response, None
                
                # Log warnings if any
                if validation_result.warnings:
                    self.logger.warning(f"⚠️ Query validation warnings in updated analysis: {validation_result.warnings}")
            except Exception as e:
                self.logger.error(f"❌ Validation error in updated analysis: {e}")
                # If validation fails, continue with the query anyway
                self.logger.info("🔍 Continuing with query despite validation error")
            
            # Get enhanced context from ChromaDB
            enhanced_context = self.context_manager.get_enhanced_context(user_query, query_analysis)
            self.logger.info(f"🔍 Enhanced context: {enhanced_context}")
            
            # Execute tools with enhanced reasoning
            tool_results = self._execute_tools_with_reasoning(user_query, query_analysis)
            
            # Generate intelligent response
            response = self._generate_intelligent_summary_with_reasoning(user_query, query_analysis, tool_results)
            
            # Add to conversation history with enhanced context
            self.context_manager.add_to_history(user_query, response, query_analysis)
            
            # Extract visualization data if present
            visualization_data = None
            visualization_tools = [
                "create_lap_time_progression", "create_position_progression", 
                "create_sector_analysis", "create_pit_stop_analysis", "create_tire_strategy"
            ]
            
            for tool_name in visualization_tools:
                if tool_name in tool_results and tool_results[tool_name].get("success"):
                    visualization_data = tool_results[tool_name]
                    self.logger.info(f"🔍 Found visualization data: {visualization_data}")
                    break
            
            return response, visualization_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in continuing with updated analysis: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}", None

    def _execute_single_tool(self, tool_name: str, tool_config: Dict[str, Any], query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool with proper error handling"""
        
        try:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                
                # Check if the tool's execute method is async
                if inspect.iscoroutinefunction(tool.execute):
                    # Handle async execution
                    try:
                        # Try to get the current event loop
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If we're already in an event loop, we need to handle this differently
                            # For now, let's try to run it in a new thread
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, tool.execute(tool_config))
                                result = future.result(timeout=30)  # 30 second timeout
                        else:
                            # If no event loop is running, we can use asyncio.run
                            result = asyncio.run(tool.execute(tool_config))
                    except RuntimeError as e:
                        if "asyncio.run() cannot be called from a running event loop" in str(e):
                            # Fallback: try to create a new event loop
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(self._run_async_tool, tool, tool_config)
                                result = future.result(timeout=30)
                        else:
                            raise e
                else:
                    # Synchronous execution
                    result = tool.execute(tool_config)
                
                # Convert ToolResult to dict if needed
                if hasattr(result, 'success'):
                    # This is a ToolResult object - convert to dict
                    result_dict = {
                        "success": result.success,
                        "data": result.data,
                        "error": getattr(result, 'error', None),
                        "sql_query": getattr(result, 'sql_query', None),
                        "sql_params": getattr(result, 'sql_params', None),
                        "execution_time": getattr(result, 'execution_time', None)
                    }
                    
                    # For visualization tools, extract key fields to top level for frontend
                    if tool_name in ["create_lap_time_progression", "create_position_progression", 
                                   "create_sector_analysis", "create_pit_stop_analysis", "create_tire_strategy"]:
                        if result.success and result.data:
                            result_dict.update({
                                "filename": result.data.get("filename"),
                                "visualization_type": result.data.get("visualization_type"),
                                "session_key": result.data.get("session_key"),
                                "total_laps": result.data.get("total_laps"),
                                "total_drivers": result.data.get("total_drivers"),
                                "drivers_included": result.data.get("drivers_included", [])
                            })
                    
                    return result_dict
                elif hasattr(result, 'content'):
                    return {"content": result.content, "success": True}
                elif isinstance(result, dict):
                    return result
                else:
                    return {"content": str(result), "success": True}
            else:
                self.logger.warning(f"🔧 Tool {tool_name} not found")
                return {"error": f"Tool {tool_name} not available"}
        except Exception as e:
            self.logger.error(f"❌ Error executing tool {tool_name}: {str(e)}")
            return {"error": f"Error executing {tool_name}: {str(e)}"}

    def _run_async_tool(self, tool, tool_config):
        """Helper method to run async tool in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(tool.execute(tool_config))
        finally:
            loop.close()

    def _validate_tool_results(self, tool_results: Dict[str, Any], query_analysis: Dict[str, Any]) -> str:
        """Validate tool results and provide feedback"""
        
        errors = []
        successes = []
        
        for tool_name, result in tool_results.items():
            if "error" in result:
                errors.append(f"{tool_name}: {result['error']}")
            else:
                successes.append(tool_name)
        
        if errors:
            return f"Errors: {', '.join(errors)}"
        else:
            return f"All tools executed successfully: {', '.join(successes)}"

    def _generate_intelligent_summary_with_reasoning(self, user_query: str, query_analysis: Dict[str, Any], tool_results: Dict[str, Any]) -> str:
        """Generate intelligent summary with enhanced reasoning"""
        
        if "error" in tool_results:
            error_msg = tool_results['error']
            
            # Provide more helpful error responses based on the error type
            if "Could not find meeting" in error_msg:
                return f"I couldn't find the race you specified in our database. Please check the race name and try again. Valid races include: Monaco Grand Prix, British Grand Prix, Bahrain Grand Prix, etc."
            elif "Could not find session" in error_msg:
                return f"I couldn't find the session type you specified. Most data is available for 'Race' sessions. Try specifying 'Race' instead of other session types."
            elif "No performance data found" in error_msg:
                query_type = query_analysis.get("query_type", "")
                if query_type == "driver_performance":
                    return f"I couldn't find performance data for the specified driver. Please check the driver name and try again. Valid drivers include: Max Verstappen, Lewis Hamilton, Charles Leclerc, etc."
                elif query_type == "team_performance":
                    return f"I couldn't find performance data for the specified team. Please check the team name and try again. Valid teams include: Red Bull, Mercedes, Ferrari, McLaren, etc."
                else:
                    return f"I couldn't find the data you requested. Please check your query and try again."
            else:
                return f"I encountered an issue processing your query. Please try a simpler query like:\n• 'lap time progression for Max Verstappen at Monaco GP'\n• 'team performance for Red Bull at Bahrain GP'"
        
        # Check if there are any successful tool results
        has_successful_results = False
        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and result.get("success"):
                has_successful_results = True
                break
        
        # If no successful results, provide a helpful error message
        if not has_successful_results:
            return "I couldn't find any data for your query. Please try a simpler query like:\n• 'lap time progression for Max Verstappen at Monaco GP'\n• 'team performance for Red Bull at Bahrain GP'\n• 'sector analysis for Charles Leclerc at Silverstone'"
        
        # Get conversation context
        conversation_context = self._get_conversation_context()
        
        # Check if visualization was generated - look for visualization tool results
        visualization_tools = [
            "create_lap_time_progression", "create_position_progression", 
            "create_sector_analysis", "create_pit_stop_analysis", "create_tire_strategy"
        ]
        
        has_visualization = False
        viz_data = None
        viz_type = None
        
        for tool_name in visualization_tools:
            if tool_name in tool_results and tool_results[tool_name].get("success"):
                has_visualization = True
                viz_data = tool_results[tool_name]
                viz_type = tool_name.replace("create_", "")
                break
        
        # Create filtered tool results for LLM
        filtered_tool_results = self._filter_tool_results_for_llm(tool_results)
        
        # Add debugging for visualization data
        if has_visualization:
            self.logger.info(f"🔍 Visualization detected: {viz_data}")
            self.logger.info(f"🔍 Filtered tool results for LLM: {filtered_tool_results}")
        
        # If visualization was generated, analyze it
        if has_visualization and viz_data:
            self.logger.info(f"🔍 Analyzing generated {viz_type} chart")
            chart_analysis = self.chart_analyzer.analyze_chart(viz_data, viz_type, user_query)
            
            # Generate base response
            try:
                summary_prompt = self._get_enhanced_visualization_prompt(user_query, query_analysis, filtered_tool_results, conversation_context, viz_type)
                messages = [SystemMessage(content=summary_prompt)]
                response = self.llm.invoke(messages)
                base_response = response.content
                
                # Combine base response with chart analysis
                combined_response = f"{base_response}\n\n📊 **Chart Analysis:**\n{chart_analysis}"
                
                return combined_response
                
            except Exception as e:
                self.logger.error(f"❌ Error in combined response generation: {e}")
                # Fallback to just chart analysis
                return f"I've generated a {viz_type} visualization for you.\n\n📊 **Chart Analysis:**\n{chart_analysis}"
        
        # For non-visualization responses, use the original logic
        query_type = query_analysis.get("query_type", "")
        
        if query_type == "visualization":
            summary_prompt = self._get_enhanced_visualization_prompt(user_query, query_analysis, filtered_tool_results, conversation_context, "data visualization")
        else:
            summary_prompt = self._get_enhanced_text_prompt(user_query, query_analysis, filtered_tool_results, conversation_context)
        
        try:
            messages = [SystemMessage(content=summary_prompt)]
            response = self.llm.invoke(messages)
            answer = response.content
            
            return answer
            
        except Exception as e:
            self.logger.error(f"❌ Error generating summary: {e}")
            
            # Check if it's a token limit error
            if "context length" in str(e) or "128000 tokens" in str(e) or "maximum context length" in str(e):
                return "Your query is too long and I can't process that much data at once. Please try a more specific query like:\n• 'lap time progression for Max Verstappen at Monaco GP'\n• 'team performance for Red Bull at Bahrain GP'\n• 'sector analysis for Charles Leclerc at Silverstone'"
            else:
                return self._generate_basic_summary(user_query, query_analysis, tool_results)

    def _filter_tool_results_for_llm(self, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter tool results to remove sensitive data before sending to LLM"""
        
        filtered_results = {}
        for key, value in tool_results.items():
            # Handle visualization tool results
            if key in ["create_lap_time_progression", "create_position_progression", 
                      "create_sector_analysis", "create_pit_stop_analysis", "create_tire_strategy"]:
                if isinstance(value, dict) and value.get("success"):
                    viz_data = value
                    filtered_results[key] = {
                        "success": viz_data.get("success"),
                        "visualization_type": viz_data.get("visualization_type", "data visualization"),
                        "filename": viz_data.get("filename"),
                        "session_key": viz_data.get("session_key"),
                        "total_laps": viz_data.get("total_laps"),
                        "total_drivers": viz_data.get("total_drivers"),
                        "drivers_included": viz_data.get("drivers_included", [])[:5],
                        "visualization_generated": True,
                        "data_summary": f"Generated {viz_data.get('visualization_type', 'data visualization')} chart with {viz_data.get('total_drivers', 0)} drivers over {viz_data.get('total_laps', 0)} laps"
                    }
                else:
                    filtered_results[key] = value
            else:
                filtered_results[key] = value
        
        return filtered_results

    def _execute_tools_with_reasoning(self, user_query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tools with enhanced reasoning and chain-of-thought"""
        
        reasoning_steps = []
        current_step = 1
        
        # Step 1: Analyze the query requirements
        reasoning_steps.append(f"Step {current_step}: Analyzing query requirements")
        current_step += 1
        
        query_type = query_analysis.get("query_type", "")
        meeting_info = query_analysis.get("meeting_info", {})
        drivers = query_analysis.get("drivers", [])
        teams = query_analysis.get("teams", [])
        
        reasoning_steps.append(f"  - Query type: {query_type}")
        reasoning_steps.append(f"  - Meeting: {meeting_info.get('name', 'Unknown')}")
        reasoning_steps.append(f"  - Drivers: {drivers}")
        reasoning_steps.append(f"  - Teams: {teams}")
        
        # Step 2: Determine required tools
        reasoning_steps.append(f"Step {current_step}: Determining required tools")
        current_step += 1
        
        required_tools = self._determine_required_tools(query_analysis)
        reasoning_steps.append(f"  - Required tools: {list(required_tools.keys())}")
        
        # Step 3: Execute tools with reasoning
        reasoning_steps.append(f"Step {current_step}: Executing tools")
        current_step += 1
        
        tool_results = {}
        meeting_key = None
        session_key = None
        
        # First, get meeting and session keys
        if "get_meeting_key" in required_tools:
            try:
                meeting_result = self._execute_single_tool("get_meeting_key", required_tools["get_meeting_key"], query_analysis)
                if meeting_result.get("success"):
                    # Extract meeting_key from ToolResult data
                    if hasattr(meeting_result, 'data') and meeting_result.data:
                        meeting_key = meeting_result.data.get("meeting_key")
                    elif isinstance(meeting_result, dict) and "data" in meeting_result:
                        meeting_key = meeting_result["data"].get("meeting_key")
                    else:
                        meeting_key = meeting_result.get("meeting_key")
                    reasoning_steps.append(f"    ✓ Retrieved meeting key: {meeting_key}")
                else:
                    reasoning_steps.append(f"    ✗ Failed to get meeting key: {meeting_result.get('error')}")
                    return {"error": f"Could not find meeting: {meeting_result.get('error')}"}
            except Exception as e:
                reasoning_steps.append(f"    ✗ Error getting meeting key: {str(e)}")
                return {"error": f"Error getting meeting key: {str(e)}"}
        
        if "get_session_key" in required_tools and meeting_key:
            try:
                session_config = required_tools["get_session_key"].copy()
                session_config["meeting_key"] = meeting_key
                session_result = self._execute_single_tool("get_session_key", session_config, query_analysis)
                if session_result.get("success"):
                    # Extract session_key from ToolResult data
                    if hasattr(session_result, 'data') and session_result.data:
                        session_key = session_result.data.get("session_key")
                    elif isinstance(session_result, dict) and "data" in session_result:
                        session_key = session_result["data"].get("session_key")
                    else:
                        session_key = session_result.get("session_key")
                    reasoning_steps.append(f"    ✓ Retrieved session key: {session_key}")
                else:
                    reasoning_steps.append(f"    ✗ Failed to get session key: {session_result.get('error')}")
                    return {"error": f"Could not find session: {session_result.get('error')}"}
            except Exception as e:
                reasoning_steps.append(f"    ✗ Error getting session key: {str(e)}")
                return {"error": f"Error getting session key: {str(e)}"}
        
        # Now execute the main tools with proper keys
        for tool_name, tool_config in required_tools.items():
            if tool_name in ["get_meeting_key", "get_session_key"]:
                continue  # Already handled
                
            reasoning_steps.append(f"  - Executing {tool_name}")
            try:
                # Update tool config with proper keys
                updated_config = tool_config.copy()
                if "meeting_key" in updated_config and updated_config["meeting_key"] == "placeholder":
                    updated_config["meeting_key"] = meeting_key
                if "session_key" in updated_config and updated_config["session_key"] == "placeholder":
                    updated_config["session_key"] = session_key
                
                result = self._execute_single_tool(tool_name, updated_config, query_analysis)
                tool_results[tool_name] = result
                
                if result.get("success"):
                    reasoning_steps.append(f"    ✓ {tool_name} completed successfully")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    reasoning_steps.append(f"    ✗ {tool_name} failed: {error_msg}")
                    
                    # Provide more helpful error messages based on the tool and error
                    if "No performance data found" in error_msg:
                        if tool_name == "get_driver_performance":
                            reasoning_steps.append(f"      → No data available for the specified driver")
                        elif tool_name == "get_team_performance":
                            reasoning_steps.append(f"      → No data available for the specified team")
                    elif "Could not find meeting" in error_msg:
                        reasoning_steps.append(f"      → The specified race may not exist in our database")
                    elif "Could not find session" in error_msg:
                        reasoning_steps.append(f"      → The specified session type may not be available")
            except Exception as e:
                reasoning_steps.append(f"    ✗ {tool_name} failed: {str(e)}")
                tool_results[tool_name] = {"error": str(e)}
        
        # Step 4: Validate results
        reasoning_steps.append(f"Step {current_step}: Validating results")
        current_step += 1
        
        validation_result = self._validate_tool_results(tool_results, query_analysis)
        reasoning_steps.append(f"  - Validation: {validation_result}")
        
        # Check if all tools failed
        failed_tools = [name for name, result in tool_results.items() if "error" in result]
        if len(failed_tools) == len(tool_results) and len(tool_results) > 0:
            reasoning_steps.append(f"  - All tools failed, providing helpful suggestions")
            # Add helpful suggestions to the tool_results
            tool_results["suggestions"] = self._generate_failure_suggestions(query_analysis, failed_tools)
        
        # Log reasoning steps
        self.logger.info("🧠 Reasoning steps:")
        for step in reasoning_steps:
            self.logger.info(f"  - {step}")
        
        return tool_results

    def _determine_required_tools(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine which tools are required based on query analysis"""
        
        query_type = query_analysis.get("query_type", "")
        meeting_info = query_analysis.get("meeting_info", {})
        drivers = query_analysis.get("drivers", [])
        teams = query_analysis.get("teams", [])
        session_type = query_analysis.get("session_type", "Race")
        
        tools_needed = {}
        
        # Get meeting and session keys first
        tools_needed["get_meeting_key"] = {
            "event_name": meeting_info.get("name", ""),  # Fixed: Changed from meeting_name to event_name
            "year": meeting_info.get("year", "")
        }
        
        tools_needed["get_session_key"] = {
            "meeting_key": "placeholder",  # Will be replaced after meeting_key is retrieved
            "session_type": session_type
        }
        
        # Map query types to specific tools - use the actual tool names from your system
        if query_type == "comparison":
            if drivers and len(drivers) >= 2:
                tools_needed["compare_drivers"] = {
                    "drivers": drivers,
                    "meeting_key": "placeholder",
                    "session_key": "placeholder"
                }
            elif teams and len(teams) >= 2:
                tools_needed["compare_teams"] = {
                    "teams": teams,
                    "meeting_key": "placeholder",
                    "session_key": "placeholder"
                }
            else:
                # Fallback to driver comparison if no specific drivers/teams mentioned
                tools_needed["compare_drivers"] = {
                    "drivers": drivers,
                    "meeting_key": "placeholder",
                    "session_key": "placeholder"
                }
                
        elif query_type == "driver_performance":
            tools_needed["get_driver_performance"] = {
                "driver_name": drivers[0] if drivers else None,
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
                
        elif query_type == "team_performance":
            tools_needed["get_team_performance"] = {
                "team_name": teams[0] if teams else None,
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
                
        elif query_type == "fastest_lap":
            tools_needed["get_fastest_lap"] = {
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
            
        elif query_type == "sector_analysis":
            # Convert drivers and teams lists to comma-separated strings
            driver_filter = ", ".join(drivers) if drivers else None
            team_filter = ", ".join(teams) if teams else None
            
            tools_needed["get_sector_analysis"] = {
                "driver_filter": driver_filter,
                "team_filter": team_filter,
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
                
        elif query_type == "position_analysis":
            # Convert drivers and teams lists to comma-separated strings
            driver_filter = ", ".join(drivers) if drivers else None
            team_filter = ", ".join(teams) if teams else None
            
            tools_needed["get_position_progression"] = {
                "driver_filter": driver_filter,
                "team_filter": team_filter,
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
                
        elif query_type == "lap_time_progression":
            # Convert drivers list to string format for visualization tool
            driver_filter = ", ".join(drivers) if drivers else None
            team_filter = ", ".join(teams) if teams else None
            
            tools_needed["create_lap_time_progression"] = {
                "driver_filter": driver_filter,
                "team_filter": team_filter,
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
            
        elif query_type == "pit_strategy":
            # Convert drivers and teams lists to comma-separated strings
            driver_filter = ", ".join(drivers) if drivers else None
            team_filter = ", ".join(teams) if teams else None
            
            tools_needed["get_pit_stop_analysis"] = {
                "driver_filter": driver_filter,
                "team_filter": team_filter,
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
                
        elif query_type == "tire_strategy":
            # Convert drivers and teams lists to comma-separated strings
            driver_filter = ", ".join(drivers) if drivers else None
            team_filter = ", ".join(teams) if teams else None
            
            tools_needed["get_tire_strategy"] = {
                "driver_filter": driver_filter,
                "team_filter": team_filter,
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
                
        elif query_type == "incident_investigation":
            tools_needed["investigate_incident"] = {
                "meeting_key": "placeholder",
                "session_key": "placeholder",
                "lap_number": query_analysis.get("lap_number")
            }
            
        elif query_type == "qualifying_results":
            tools_needed["get_qualifying_results"] = {
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
            
        elif query_type == "race_results":
            tools_needed["get_race_results"] = {
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
            
        elif query_type == "visualization":
            # For visualization, we need to determine the specific type
            original_query = query_analysis.get("original_query", "").lower()
            self.logger.info(f"🔍 Visualization tool selection - original_query: '{original_query}'")
            if "sector" in original_query:
                # Convert drivers list to string format for visualization tool
                driver_filter = ", ".join(drivers) if drivers else None
                team_filter = ", ".join(teams) if teams else None
                
                self.logger.info(f"🔍 Selecting create_sector_analysis tool")
                tools_needed["create_sector_analysis"] = {
                    "driver_filter": driver_filter,
                    "team_filter": team_filter,
                    "meeting_key": "placeholder",
                    "session_key": "placeholder"
                }
            elif "position" in original_query:
                # Convert drivers list to string format for visualization tool
                driver_filter = ", ".join(drivers) if drivers else None
                team_filter = ", ".join(teams) if teams else None
                
                tools_needed["create_position_progression"] = {
                    "driver_filter": driver_filter,
                    "team_filter": team_filter,
                    "meeting_key": "placeholder",
                    "session_key": "placeholder"
                }
            elif "lap" in original_query or "time" in original_query:
                # Convert drivers list to string format for visualization tool
                driver_filter = ", ".join(drivers) if drivers else None
                team_filter = ", ".join(teams) if teams else None
                
                tools_needed["create_lap_time_progression"] = {
                    "driver_filter": driver_filter,
                    "team_filter": team_filter,
                    "meeting_key": "placeholder",
                    "session_key": "placeholder"
                }
            elif "pit" in original_query:
                # Convert drivers list to string format for visualization tool
                driver_filter = ", ".join(drivers) if drivers else None
                team_filter = ", ".join(teams) if teams else None
                
                tools_needed["create_pit_stop_analysis"] = {
                    "driver_filter": driver_filter,
                    "team_filter": team_filter,
                    "meeting_key": "placeholder",
                    "session_key": "placeholder"
                }
            elif "tire" in original_query or "tyre" in original_query:
                # Convert drivers list to string format for visualization tool
                driver_filter = ", ".join(drivers) if drivers else None
                team_filter = ", ".join(teams) if teams else None
                
                tools_needed["create_tire_strategy"] = {
                    "driver_filter": driver_filter,
                    "team_filter": team_filter,
                    "meeting_key": "placeholder",
                    "session_key": "placeholder"
                }
            else:
                # Fallback to general lap time progression for visualization
                # Convert drivers list to string format for visualization tool
                driver_filter = ", ".join(drivers) if drivers else None
                team_filter = ", ".join(teams) if teams else None
                
                self.logger.info(f"🔍 Falling back to create_lap_time_progression tool")
                tools_needed["create_lap_time_progression"] = {
                    "driver_filter": driver_filter,
                    "team_filter": team_filter,
                    "meeting_key": "placeholder",
                    "session_key": "placeholder"
                }
        elif query_type == "chart_analysis":
            # For chart analysis, we need to get the last visualization from context
            # This will be handled by the reasoning engine
            tools_needed["analyze_last_chart"] = {
                "analysis_type": "comprehensive",
                "user_query": query_analysis.get("original_query", "")
            }
        else:
            # Default to race results
            tools_needed["get_race_results"] = {
                "meeting_key": "placeholder",
                "session_key": "placeholder"
            }
        
        return tools_needed

    def _get_enhanced_visualization_prompt(self, user_query: str, query_analysis: Dict[str, Any], filtered_tool_results: Dict[str, Any], conversation_context: str, viz_type: str) -> str:
        """Generate enhanced prompt for visualization queries with direct responses"""
        
        query_type = query_analysis.get("query_type", "")
        
        prompt = f"""You are a Formula 1 assistant. Answer the user's question directly and concisely.

User Question: {user_query}

Query Type: {query_type}

Available Data: {filtered_tool_results}

Instructions:
- Answer the question directly without unnecessary introduction
- Be concise and to the point
- Focus on the key information the user asked for
- Avoid academic language or lengthy explanations
- If no data is available, say so clearly
- Use bullet points for multiple items when helpful
- For visualizations that were successfully generated, acknowledge the chart and provide brief insights about what the data shows
- If the visualization shows limited data (e.g., only one driver), explain why (e.g., "The chart shows data for [driver] only, as [other driver] retired early")
- Keep responses natural and conversational
- Don't mention technical details like filenames or session keys
- NEVER use third-person meta-commentary like "You could respond with..." or "Here's what you might say..."
- ALWAYS respond as if you are directly answering the user

Provide a direct answer:"""
        
        return prompt

    def _get_enhanced_text_prompt(self, user_query: str, query_analysis: Dict[str, Any], filtered_tool_results: Dict[str, Any], conversation_context: str) -> str:
        """Generate enhanced text prompt for direct, concise responses"""
        
        query_type = query_analysis.get("query_type", "")
        
        # Create a more direct, concise prompt
        prompt = f"""You are a Formula 1 assistant. Answer the user's question directly and concisely.

User Question: {user_query}

Query Type: {query_type}

Available Data: {filtered_tool_results}

Instructions:
- Answer the question directly without unnecessary introduction
- Be concise and to the point
- Focus on the key information the user asked for
- Avoid academic language or lengthy explanations
- If no data is available, say so clearly
- Use bullet points for multiple items when helpful
- Format all lap times in MM:SS.mmm format (e.g., 74.089 seconds should be displayed as 01:14.089)
- For average lap, best lap, and worst lap times, always use MM:SS.mmm format
- Consistency can remain in seconds as it's a variance measure
- NEVER use third-person meta-commentary like "You could respond with..." or "Here's what you might say..."
- ALWAYS respond as if you are directly answering the user

For pit stop analysis:
- Provide insights and patterns rather than just listing all pit stops
- Highlight key strategies, timing decisions, and notable performances
- Mention fastest/slowest stops, strategic decisions, and team patterns
- Focus on the most interesting or important aspects of the pit stop strategy

Provide a direct answer:"""
        
        return prompt

    def _extract_team_names(self, query: str) -> List[str]:
        """Extract team names from query"""
        team_mappings = {
            "Mercedes": ["mercedes", "mercedes team", "mercedes amg"],
            "Ferrari": ["ferrari", "ferrari team", "scuderia ferrari"],
            "Red Bull Racing": ["red bull", "red bull racing", "red bull team", "redbull"],
            "McLaren": ["mclaren", "mclaren team", "mclaren racing"],
            "Aston Martin": ["aston martin", "aston martin team", "aston"],
            "Alpine": ["alpine", "alpine team", "alpine racing"],
            "Williams": ["williams", "williams team", "williams racing"],
            "AlphaTauri": ["alphatauri", "alpha tauri", "alpha tauri team"],
            "Alfa Romeo": ["alfa romeo", "alfa", "alfa romeo team"],
            "Haas": ["haas", "haas team", "haas f1"],
            "Sauber": ["sauber", "sauber team", "sauber racing"],
            "Racing Bulls": ["racing bulls", "racing bulls team", "bulls"]
        }
        
        query_lower = query.lower()
        found_teams = []
        
        for full_name, aliases in team_mappings.items():
            for alias in aliases:
                if alias in query_lower:
                    if full_name not in found_teams:
                        found_teams.append(full_name)
                    break
        
        return found_teams

    def _check_for_ambiguous_query(self, user_query: str, query_analysis: Dict[str, Any]) -> Optional[str]:
        """Check if query is ambiguous and needs clarification about visualization vs summary"""
        query_lower = user_query.lower()
        self.logger.info(f"🔍 Checking for ambiguous query: {user_query}")
        
        # Skip ambiguous check for specific query types that are clearly one type
        if query_analysis.get("query_type") in ["race_results", "qualifying_results", "fastest_lap", "incident_investigation"]:
            self.logger.info(f"🔍 Skipping ambiguous check for specific query type: {query_analysis.get('query_type')}")
            return None
        
        # Force pit_strategy queries to be treated as ambiguous (they can be text or visualization)
        if query_analysis.get("query_type") == "pit_strategy":
            self.logger.info(f"🔍 Forcing pit_strategy to be treated as ambiguous")
            # Don't return None - continue with ambiguous check
        
        # First, check for CLEAR visualization keywords (should NOT trigger clarification)
        clear_viz_keywords = [
            "visualize", "visualization", "show me a", "create a", "plot", "graph", "chart",
            "display", "show the", "create a chart", "create a graph", "create a plot",
            "show me the chart", "show me the graph", "show me the plot", "visualize the",
            "create visualization", "show visualization", "display chart", "display graph",
            "show me a chart", "show me a graph", "create a visualization", "show me a visualization"
        ]

        has_clear_viz = any(keyword in query_lower for keyword in clear_viz_keywords)
        if has_clear_viz:
            # This is clearly a visualization request, don't ask for clarification
            query_analysis["query_type"] = "visualization"
            return None
        
        # Check for CLEAR text analysis keywords (should NOT trigger clarification)
        clear_text_keywords = [
            "summarize", "summary", "tell me", "what were", "give me the data",
            "describe", "explain", "analyze", "report", "details", "comparison", "between",
            "progression", "changes", "movement", "development", "evolution", "history",
            "text analysis", "text summary", "data summary", "performance summary"
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
            "lap time", "lap analysis between", "lap comparison between",
            
            # Position related  
            "position progression", "position changes", "position analysis",
            "how did positions change", "position comparison", "position chart",
            "position changes for", "position analysis between", "position comparison between",
            
            # Sector related
            "sector analysis", "sector performance", "sector comparison",
            "sector times", "sector breakdown", "sector chart",
            "sector analysis between", "sector comparison between", "sector times for",
            
            # Pit stop related
            "pit stop analysis", "pit stop performance", "pit stop timing",
            "pit stop strategy", "pit stop comparison", "pit stop chart",
            "pit stop analysis between", "pit stop comparison between",
            
            # Tire strategy related
            "tire strategy", "tyre strategy", "tire analysis", "tyre analysis",
            "tire performance", "tyre performance", "tire comparison",
            "tire strategy between", "tire comparison between", "tire compounds between",
            
            # General comparison patterns (only if not clearly visualization/text)
            "between", "versus", "vs", "compare", "comparison",
            
            # General analysis patterns (only if not clearly visualization/text)
            "analysis", "performance", "data", "stats", "statistics"
        ]
        
        has_ambiguous_pattern = any(pattern in query_lower for pattern in ambiguous_patterns)
        
        if has_ambiguous_pattern:
            # Store context for when user responds (don't require drivers/teams)
            drivers = query_analysis.get("drivers", [])
            teams = self._extract_team_names(user_query)
            
            # Store context for when user responds
            self.context_manager.set_clarification_context({
                "meeting_info": query_analysis["meeting_info"],
                "session_type": query_analysis["session_type"],
                "drivers": drivers,
                "teams": teams,
                "original_query": user_query,
                "query_type": query_analysis.get("query_type", "")
            })
            
            # Generate a direct, helpful clarification response
            return "I can help you with that! Would you like me to create a visualization chart, or would you prefer a text summary of the data?"
        
        return None

    def _handle_clarification_response(self, user_query: str, query_analysis: Dict[str, Any]) -> Optional[str]:
        """Handle user's response to clarification about visualization vs summary"""
        
        query_lower = user_query.lower()
        clarification_context = self.context_manager.get_clarification_context()
        
        if not clarification_context:
            self.logger.warning("No clarification context found")
            return None
        
        try:
            # Use LLM to interpret the clarification response
            interpretation_prompt = f"""
The user was asked whether they wanted a visualization or text summary. They responded: "{user_query}"

Please classify their response as either:
- "visualization" (if they want a chart/graph)
- "text_analysis" (if they want a text summary/analysis)
- "unclear" (if the response is ambiguous)

Respond with ONLY the classification.
"""
            
            response = self.llm.invoke(interpretation_prompt)
            interpretation = response.content.strip().lower()
            
            if "visualization" in interpretation or "chart" in interpretation or "graph" in interpretation:
                # Update query analysis with stored context
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
                elif "tire" in original_query or "tyre" in original_query:
                    query_analysis["visualization_type"] = "tire_strategy"
                elif "lap" in original_query or "time" in original_query:
                    query_analysis["visualization_type"] = "lap_time_progression"
                else:
                    # Default to lap time progression for general comparisons
                    query_analysis["visualization_type"] = "lap_time_progression"
                
                self.context_manager.clear_clarification_context()
                return None  # Continue with execution
                
            elif "text_analysis" in interpretation or "text" in interpretation or "summary" in interpretation:
                # Update query analysis with stored context
                query_analysis.update(clarification_context)
                
                # Determine appropriate text analysis type based on original query
                original_query = clarification_context.get("original_query", "").lower()
                
                if "compare" in original_query or "vs" in original_query or "versus" in original_query:
                    query_analysis["query_type"] = "comparison"
                    # Restore the original teams and drivers
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
                elif "tire" in original_query or "tyre" in original_query:
                    query_analysis["query_type"] = "tire_strategy"
                elif "lap" in original_query or "time" in original_query:
                    query_analysis["query_type"] = "lap_time_progression"
                else:
                    # Preserve the original query type from the clarification context
                    original_query_type = clarification_context.get("query_type", "driver_performance")
                    query_analysis["query_type"] = original_query_type
                    self.logger.info(f"🔍 Preserved original query type: {original_query_type} from clarification context")
                
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
                elif "tire" in original_query or "tyre" in original_query:
                    query_analysis["visualization_type"] = "tire_strategy"
                elif "lap" in original_query or "time" in original_query:
                    query_analysis["visualization_type"] = "lap_time_progression"
                else:
                    query_analysis["visualization_type"] = "lap_time_progression"
                
                self.context_manager.clear_clarification_context()
                return None
            elif any(keyword in query_lower for keyword in text_keywords):
                self.logger.info(f"🔍 Text summary requested, clarification context: {clarification_context}")
                query_analysis.update(clarification_context)
                
                # Check if the original query was a comparison
                original_query = clarification_context.get("original_query", "").lower()
                if "compare" in original_query or "vs" in original_query or "versus" in original_query:
                    query_analysis["query_type"] = "comparison"
                    # Restore the original teams and drivers
                    if "teams" in clarification_context:
                        query_analysis["teams"] = clarification_context["teams"]
                    if "drivers" in clarification_context:
                        query_analysis["drivers"] = clarification_context["drivers"]
                else:
                    # Preserve the original query type from the clarification context
                    original_query_type = clarification_context.get("query_type", "driver_performance")
                    query_analysis["query_type"] = original_query_type
                    self.logger.info(f"🔍 Preserved original query type: {original_query_type} from clarification context")
                
                self.context_manager.clear_clarification_context()
                return None
            else:
                return "I didn't quite understand. Would you like a visualization chart or a text summary?"

    def _generate_failure_suggestions(self, query_analysis: Dict[str, Any], failed_tools: List[str]) -> List[str]:
        """Generate helpful suggestions when tools fail"""
        suggestions = []
        query_type = query_analysis.get("query_type", "")
        drivers = query_analysis.get("drivers", [])
        teams = query_analysis.get("teams", [])
        meeting_info = query_analysis.get("meeting_info", {})
        
        # General suggestions
        suggestions.append("Try using 'Race' session type instead of other sessions")
        suggestions.append("Check that driver/team names are spelled correctly")
        suggestions.append("Ensure the race name is correct (e.g., 'Monaco Grand Prix' not 'Monaco GP')")
        
        # Type-specific suggestions
        if query_type == "driver_performance":
            if not drivers:
                suggestions.append("Try specifying a driver: 'driver performance for Max Verstappen at Monaco GP'")
            else:
                suggestions.append("Try a different driver: 'driver performance for Lewis Hamilton at British GP'")
        elif query_type == "team_performance":
            if not teams:
                suggestions.append("Try specifying a team: 'team performance for Red Bull at Bahrain GP'")
            else:
                suggestions.append("Try a different team: 'team performance for Ferrari at Silverstone'")
        elif query_type == "lap_time_progression":
            suggestions.append("Try: 'lap time progression for Max Verstappen at Monaco GP'")
        elif query_type == "sector_analysis":
            suggestions.append("Try: 'sector analysis for Charles Leclerc at Monaco GP'")
        elif query_type == "position_analysis":
            suggestions.append("Try: 'position progression for Lando Norris at Silverstone'")
        elif query_type == "pit_strategy":
            suggestions.append("Try: 'pit stop analysis for Max Verstappen at Monaco GP'")
        elif query_type == "tire_strategy":
            suggestions.append("Try: 'tire strategy for Lewis Hamilton at Bahrain GP'")
        
        # Meeting-specific suggestions
        if meeting_info and meeting_info.get("name"):
            suggestions.append(f"Try a different race: 'British Grand Prix', 'Bahrain Grand Prix', 'Australian Grand Prix'")
        
        return suggestions

    def _get_conversation_context(self) -> str:
        """Get recent conversation context for the LLM (like your notebook)"""
        if not self.context_manager.conversation_history:
            return ""
        
        context = "Recent conversation context:\n"
        for entry in self.context_manager.conversation_history[-2:]:  # Last 2 exchanges
            context += f"User: {entry.get('user_message', entry.get('message', ''))}\n"
            # Truncate response to avoid too much context
            response_text = entry.get('bot_response', entry.get('response', ''))
            response_preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            context += f"Assistant: {response_preview}\n\n"
        
        return context
    
    def _generate_token_limit_error_response(self, user_query: str, query_analysis: Dict[str, Any]) -> str:
        """Generate a helpful response when the token limit is exceeded"""
        
        query_type = query_analysis.get("query_type", "")
        meeting_info = query_analysis.get("meeting_info", {})
        drivers = query_analysis.get("drivers", [])
        teams = query_analysis.get("teams", [])
        
        # Build helpful suggestions based on the query type
        suggestions = []
        
        if query_type in ["sector_analysis", "visualization"]:
            if drivers:
                suggestions.append(f"focus on a specific driver like '{drivers[0]}'")
            suggestions.append("limit to a specific range of laps (e.g., 'laps 1-20')")
            suggestions.append("focus on a specific sector (e.g., 'sector 1 analysis')")
        elif query_type in ["position_analysis", "lap_time_progression"]:
            if drivers:
                suggestions.append(f"compare just two drivers like '{drivers[0]} vs {drivers[1] if len(drivers) > 1 else 'another driver'}'")
            suggestions.append("focus on a specific time period in the race")
        elif query_type in ["pit_strategy", "tire_strategy"]:
            if drivers:
                suggestions.append(f"focus on a specific driver like '{drivers[0]}'")
            if teams:
                suggestions.append(f"focus on a specific team like '{teams[0]}'")
        else:
            suggestions.append("focus on a specific driver or team")
            suggestions.append("limit to a specific time period")
        
        meeting_name = meeting_info.get("name", "this race")
        
        response = f"I found too much data for {meeting_name} to process at once. The dataset is quite large and exceeded my processing limits.\n\n"
        response += "Can you try a more specific query like:\n"
        
        for suggestion in suggestions[:3]:  # Limit to 3 suggestions
            response += f"• {suggestion}\n"
        
        response += "\nThis will help me provide you with more focused and useful information!"
        
        return response

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