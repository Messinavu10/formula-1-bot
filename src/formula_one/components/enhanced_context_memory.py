from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.config_entity import DatabaseConfig
from src.formula_one.components.advanced_memory import VectorMemoryManager

class EnhancedContextManager(BaseComponent):
    """Enhanced context manager with vector memory integration"""
    
    def __init__(self, config, db_config: DatabaseConfig):
        super().__init__(config, db_config)
        
        # Initialize vector memory manager
        self.vector_memory = VectorMemoryManager(config, db_config)
        
        # Legacy conversation history (for backward compatibility)
        self.conversation_history = []
        self.last_clarification_context = None
        self.last_meeting_context = None
        
        self.logger.info("🧠 Enhanced Context Manager initialized")
    
    def add_to_history(self, message: str, response: str, query_analysis: Dict[str, Any] = None):
        """Add to conversation history with vector memory"""
        # Legacy history (for backward compatibility)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response
        }
        self.conversation_history.append(entry)
        
        # Always store in vector memory (create basic analysis if not provided)
        if query_analysis is None:
            # Create basic query analysis for vector storage
            query_analysis = {
                "query_type": "general",
                "meeting_info": {"name": None, "year": 2025},
                "session_type": "Race",
                "drivers": [],
                "teams": []
            }
        
        self.vector_memory.store_exchange(message, response, query_analysis)
        
        self.logger.info(f"🔍 Added to enhanced history: {entry}")
    
    def get_enhanced_context(self, user_query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced context using vector memory"""
        try:
            # Check for topic change
            if self.vector_memory.detect_topic_change(query_analysis):
                self.vector_memory.clear_context_for_new_topic()
            
            # Retrieve relevant context
            context = self.vector_memory.retrieve_relevant_context(user_query, query_analysis)
            
            # Add legacy context for backward compatibility
            context["legacy_history"] = self.conversation_history[-3:] if self.conversation_history else []
            
            self.logger.info(f"�� Retrieved enhanced context: {len(context.get('relevant_exchanges', []))} exchanges")
            
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Error getting enhanced context: {e}")
            # Fallback to legacy context
            return {
                "legacy_history": self.conversation_history[-3:] if self.conversation_history else [],
                "conversation_summary": {},
                "entity_memory": {},
                "context_cache": []
            }
    
    def generate_context_prompt(self, context: Dict[str, Any]) -> str:
        """Generate a context prompt for the LLM"""
        prompt_parts = []
        
        # Add conversation summary
        summary = context.get("conversation_summary", {})
        if summary.get("current_race"):
            prompt_parts.append(f"Current conversation context: We're discussing the {summary['current_race']}")
            if summary.get("current_session"):
                prompt_parts.append(f"Session: {summary['current_session']}")
            if summary.get("active_drivers"):
                prompt_parts.append(f"Active drivers: {', '.join(summary['active_drivers'])}")
            if summary.get("active_teams"):
                prompt_parts.append(f"Active teams: {', '.join(summary['active_teams'])}")
        
        # Add relevant exchanges
        relevant_exchanges = context.get("relevant_exchanges", [])
        if relevant_exchanges:
            prompt_parts.append("\nRelevant previous exchanges:")
            for i, exchange in enumerate(relevant_exchanges[:2], 1):
                prompt_parts.append(f"{i}. {exchange['exchange'][:200]}...")
        
        # Add entity matches
        entity_matches = context.get("entity_matches", [])
        if entity_matches:
            prompt_parts.append("\nEntity-specific context:")
            for match in entity_matches[:2]:
                prompt_parts.append(f"- {match['exchange'][:150]}...")
        
        if prompt_parts:
            return "\n".join(prompt_parts)
        else:
            return ""
    
    # Legacy methods for backward compatibility
    def is_clarification_response(self, user_query: str) -> bool:
        """Check if this is a clarification response"""
        if not self.conversation_history:
            return False
        
        last_response = self.conversation_history[-1]["response"]
        
        clarification_patterns = [
            # Visualization/text clarification patterns
            "Would you like me to create a visualization chart",
            "Would you prefer a text summary",
            "Would you like a visualization chart or a text summary",
            "I can help you with that! Would you like me to create",
            "Would you like a visualization or a text summary",
            
            # Race clarification patterns (FIXED)
            "I couldn't identify which race",
            "Could you please specify the race",
            "Could you please specify which race",  # ADD THIS
            "which race you're asking about",
            "which race you're referring to",      # ADD THIS
            "specify which race",                  # ADD THIS
            "specify the race",
            "Bahrain Grand Prix", "Miami Grand Prix", "Chinese Grand Prix",
            "Could you please specify",
            "I couldn't identify which",
            "please specify which race",           # ADD THIS
            "please specify the race",             # ADD THIS
            "which race",                         # ADD THIS
            "specify the race",                   # ADD THIS
            "referring to"                        # ADD THIS
        ]

        is_clarification = any(pattern in last_response for pattern in clarification_patterns)
        self.logger.info(f"🔍 Checking clarification response for: '{user_query}'")
        self.logger.info(f"🔍 Last response: '{last_response[:100]}...'")
        self.logger.info(f"🔍 Is clarification response: {is_clarification}")
        
        return is_clarification
    
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
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self.vector_memory.get_memory_stats()