from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.config_entity import DatabaseConfig

class EnhancedContextManager(BaseComponent):
    """Enhanced context manager with vector memory integration"""
    
    def __init__(self, config, db_config: DatabaseConfig):
        super().__init__(config, db_config)
        
        # Conversation history
        self.conversation_history = []
        self.last_clarification_context = None
        self.last_meeting_context = None
        
        self.logger.info("🧠 Enhanced Context Manager initialized")
    
    def add_to_history(self, message: str, response: str, query_analysis: Dict[str, Any] = None):
        """Add to conversation history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response
        }
        self.conversation_history.append(entry)
        
        self.logger.info(f"🔍 Added to history: {entry}")
    
    def get_enhanced_context(self, user_query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get conversation context"""
        # Return recent conversation history
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
        
        # Don't treat very short queries as clarifications
        if len(user_query.strip()) <= 3:
            return False
        
        last_response = self.conversation_history[-1]["response"]
        
        # More specific clarification patterns that clearly indicate a request for clarification
        clarification_patterns = [
            # Race clarification patterns - very specific
            "Could you please specify which race",
            "Could you please specify the race",
            "which race you're asking about",
            "which race you're referring to",
            "specify which race",
            "specify the race",
            "please specify which race",
            "please specify the race",
            "which race",
            "referring to",
            
            # Visualization/text clarification patterns - very specific
            "Would you like me to create a visualization chart",
            "Would you prefer a text summary",
            "Would you like a visualization chart or a text summary",
            "Would you like a visualization or a text summary"
        ]

        is_clarification = any(pattern in last_response for pattern in clarification_patterns)
        self.logger.info(f"🔍 Checking clarification response for: '{user_query}'")
        self.logger.info(f"🔍 Last response: '{last_response[:100]}...'")
        self.logger.info(f"🔍 Is clarification response: {is_clarification}")
        
        # Debug: show which pattern matched
        if is_clarification:
            for pattern in clarification_patterns:
                if pattern in last_response:
                    self.logger.info(f"🔍 Matched clarification pattern: '{pattern}'")
                    break
        
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