import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import time
import os

from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.config_entity import DatabaseConfig
from src.formula_one.entity.mcp_config_entity import MCPConfig

class IntentAnalyzer(BaseComponent):
    """Analyze user query intent and extract information"""
    
    def __init__(self, config, db_config: DatabaseConfig):
        super().__init__(config, db_config)
    
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
        
        return {
            "meeting_info": meeting_info,
            "session_type": session_type,
            "query_type": query_type,
            "drivers": self._extract_driver_names(user_query)
        }
    
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
            'great britain': 'British Grand Prix'
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
        """Determine query type from query"""
        if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            return "comparison"
        elif any(word in query_lower for word in ["performance", "how did", "analysis"]):
            return "performance"
        elif any(word in query_lower for word in ["fastest lap", "best lap", "quickest"]):
            return "fastest_lap"
        elif any(word in query_lower for word in ["pit stop", "strategy", "stops", "stints"]):
            return "strategy"
        elif any(word in query_lower for word in ["qualifying", "quali", "grid", "pole"]):
            return "qualifying_results"
        else:
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
        
        for full_name, aliases in driver_mappings.items():
            for alias in aliases:
                if alias in query_lower:
                    if full_name not in found_drivers:
                        found_drivers.append(full_name)
                    break
        
        return found_drivers[:2]

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
        """Process user query and generate response"""
        start_time = time.time()
        
        #self.logger.info(f"Processing query: {user_query}")
        
        # Enhanced clarification handling (like your notebook)
        if self.context_manager.conversation_history:
            last_response = self.context_manager.conversation_history[-1]["response"]
            
            # Check if we just asked for clarification between Sprint and main race
            if "Which one would you like to know about?" in last_response and "Sprint" in last_response:
                #self.logger.info("📝 Detected clarification response for Sprint vs Main race")
                
                # Extract the original meeting from the last user query
                last_user_query = self.context_manager.conversation_history[-1]["message"]
                meeting_info = self.intent_analyzer._extract_meeting_info(last_user_query)
                
                if meeting_info["name"]:
                    # Determine session type from current response
                    query_lower = user_query.lower()
                    
                    if any(word in query_lower for word in ["main", "grand prix", "gp", "main race"]):
                        session_type = "Race"
                    elif any(word in query_lower for word in ["sprint", "sprint race"]):
                        session_type = "Sprint"
                    else:
                        # Default based on keywords
                        session_type = "Sprint" if "sprint" in query_lower else "Race"
                    
                    query_analysis = {
                        "meeting_info": meeting_info,
                        "session_type": session_type,
                        "query_type": "race_results",
                        "drivers": []
                    }
                    #self.logger.info(f"📝 Using clarification context: {meeting_info['name']} - {session_type}")
                else:
                    query_analysis = self.intent_analyzer.analyze_query_intent(user_query, self.context_manager.conversation_history)
            else:
                query_analysis = self.intent_analyzer.analyze_query_intent(user_query, self.context_manager.conversation_history)
        else:
            query_analysis = self.intent_analyzer.analyze_query_intent(user_query, self.context_manager.conversation_history)
        
        #self.logger.info(f"Query intent: {query_analysis}")
        
        # Store meeting context for future queries (like your notebook)
        if query_analysis["meeting_info"]["name"]:
            self.context_manager.set_last_meeting_context(query_analysis["meeting_info"])
        
        # Check if we need to ask for clarification
        clarification = self._check_for_clarification_needed(user_query, query_analysis)
        if clarification:
            self.context_manager.add_to_history(user_query, clarification)
            return clarification
        
        # Execute tools based on analysis
        tool_results = self._execute_tools_for_query(user_query, query_analysis)
        
        # Generate intelligent summary
        answer = self._generate_intelligent_summary(user_query, query_analysis, tool_results)
        
        # Add to conversation history
        self.context_manager.add_to_history(user_query, answer)
        
        #execution_time = time.time() - start_time
        #self.logger.info(f"Query processed in {execution_time:.3f}s")
        
        return answer
    
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
                # Try alternative session types if the requested one fails (like your notebook)
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
            
            if query_type == "comparison":
                if len(drivers) >= 2:
                    tool_results["driver_comparison"] = self.http_client.call_tool("compare_drivers", {
                        "session_key": session_key,
                        "driver1": drivers[0],
                        "driver2": drivers[1],
                        "comparison_metrics": ["all"]
                    })
                    if "sql_query" in tool_results["driver_comparison"]:
                        sql_queries.append({"tool": "compare_drivers", "query": tool_results["driver_comparison"]["sql_query"], "params": tool_results["driver_comparison"].get("sql_params", {})})
                else:
                    tool_results["error"] = f"Need two drivers to compare. Found: {drivers}"
            
            elif query_type == "performance":
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
            
            elif query_type == "fastest_lap":
                tool_results["fastest_lap"] = self.http_client.call_tool("get_fastest_lap", {
                    "session_key": session_key
                })
                if "sql_query" in tool_results["fastest_lap"]:
                    sql_queries.append({"tool": "get_fastest_lap", "query": tool_results["fastest_lap"]["sql_query"], "params": tool_results["fastest_lap"].get("sql_params", {})})
            
            elif query_type == "strategy":
                tool_results["pit_stop_analysis"] = self.http_client.call_tool("get_pit_stop_analysis", {
                    "session_key": session_key,
                    "analysis_type": "all"
                })
                if "sql_query" in tool_results["pit_stop_analysis"]:
                    sql_queries.append({"tool": "get_pit_stop_analysis", "query": tool_results["pit_stop_analysis"]["sql_query"], "params": tool_results["pit_stop_analysis"].get("sql_params", {})})
            
            elif query_type == "qualifying_results":
                tool_results["qualifying_results"] = self.http_client.call_tool("get_qualifying_results", {
                    "session_key": session_key,
                    "result_type": "top_10"
                })
                if "sql_query" in tool_results["qualifying_results"]:
                    sql_queries.append({"tool": "get_qualifying_results", "query": tool_results["qualifying_results"]["sql_query"], "params": tool_results["qualifying_results"].get("sql_params", {})})
            
            else:  # race_results
                tool_results["race_results"] = self.http_client.call_tool("get_race_results", {
                    "session_key": session_key,
                    "result_type": "podium",
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
    
    def _generate_intelligent_summary(self, user_query: str, query_analysis: Dict[str, Any], tool_results: Dict[str, Any]) -> str:
        """Generate an intelligent text summary using LLM"""
        
        if "error" in tool_results:
            return f"I apologize, but I encountered an error: {tool_results['error']}"
        
        # Get conversation context (like your notebook)
        conversation_context = self._get_conversation_context()
        
        # Prepare context for the LLM
        context_data = {
            "query": user_query,
            "race": query_analysis["meeting_info"]["name"],
            "year": query_analysis["meeting_info"]["year"],
            "session_type": query_analysis["session_type"],
            "query_type": query_analysis["query_type"],
            "tool_results": tool_results
        }
        
        # Create a prompt for generating the summary (like your notebook)
        summary_prompt = f"""{conversation_context}You are an expert F1 analyst. Based on the following data, provide a comprehensive and engaging summary that directly answers the user's question.

User Query: {user_query}
Race: {context_data['race']} {context_data['year']}
Session: {context_data['session_type']}

Data Retrieved:
{json.dumps(tool_results, indent=2)}

Please provide a response that:
1. Directly answers the user's question in 2-3 sentences maximum
2. Includes specific numbers, times, and positions
3. Uses clear, factual language (avoid flowery descriptions)
4. Converts lap times from seconds to MM:SS.mmm format (e.g., 93.614 seconds becomes 01:33.614)
5. Mentions significant gaps only if noteworthy

IMPORTANT: 
- Keep responses concise and to the point
- Avoid phrases like "thrilling", "spectacular", "vibrant skies", "edge of their seats"
- Focus on facts: who, what times, what positions
- Maximum 3-4 sentences per response
- When displaying lap times, use MM:SS.mmm format

IMPORTANT: When displaying lap times, use the format MM:SS.mmm (e.g., 01:33.614, 01:30.000, 01:35.123). Do NOT mix words and numbers for times.

Format the response as flowing text with natural paragraphs, not as raw data dumps.
"""
        
        try:
            messages = [SystemMessage(content=summary_prompt)]
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Add SQL query information if available
            # if "sql_queries" in tool_results:
            #     answer += "\n\n" + "="*60 + "\n"
            #     answer += "�� **SQL Queries Executed:**\n"
            #     answer += "="*60 + "\n"
            #     for i, query_info in enumerate(tool_results["sql_queries"], 1):
            #         answer += f"\n**Query {i} ({query_info['tool']}):**\n"
            #         answer += f"```sql\n{query_info['query']}\n```\n"
            #         if query_info.get('params'):
            #             answer += f"**Parameters:** {query_info['params']}\n"
            
            return answer
            
        except Exception as e:
            self.logger.error(f"❌ Error generating summary: {e}")
            # Fallback to basic summary
            return self._generate_basic_summary(user_query, query_analysis, tool_results)
    
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