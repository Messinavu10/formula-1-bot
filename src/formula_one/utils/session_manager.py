import uuid
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import threading

@dataclass
class SessionData:
    """Data structure for session information"""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    max_history_length: int = 50
    max_session_age_hours: int = 24

class SessionManager:
    """Manages user sessions with proper isolation and cleanup"""
    
    def __init__(self, cleanup_interval: int = 300, max_sessions: int = 1000):
        self.sessions: Dict[str, SessionData] = {}
        self.cleanup_interval = cleanup_interval  # seconds
        self.max_sessions = max_sessions
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the background cleanup task"""
        def cleanup_loop():
            while True:
                try:
                    self._cleanup_expired_sessions()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    self.logger.error(f"Error in session cleanup: {e}")
                    time.sleep(60)  # Wait before retrying
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new session and return session ID"""
        with self._lock:
            # Clean up old sessions if we're at capacity
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_expired_sessions(force=True)
            
            session_id = str(uuid.uuid4())
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id
            )
            
            self.sessions[session_id] = session_data
            self.logger.info(f"Created new session: {session_id} for user: {user_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session and session.is_active:
                session.last_activity = datetime.now()
                return session
            return None
    
    def update_session_activity(self, session_id: str):
        """Update session last activity timestamp"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_activity = datetime.now()
    
    def add_conversation_entry(self, session_id: str, user_message: str, 
                              bot_response: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a conversation entry to the session"""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            entry = {
                'timestamp': datetime.now().isoformat(),
                'user_message': user_message,
                'bot_response': bot_response,
                'metadata': metadata or {}
            }
            
            session.conversation_history.append(entry)
            
            # Trim history if it exceeds max length
            if len(session.conversation_history) > session.max_history_length:
                session.conversation_history = session.conversation_history[-session.max_history_length:]
            
            session.last_activity = datetime.now()
            return True
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        return session.conversation_history if session else []
    
    def set_context_data(self, session_id: str, key: str, value: Any):
        """Set context data for a session"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.context_data[key] = value
                session.last_activity = datetime.now()
    
    def get_context_data(self, session_id: str, key: str, default: Any = None) -> Any:
        """Get context data for a session"""
        session = self.get_session(session_id)
        if session:
            return session.context_data.get(key, default)
        return default
    
    def end_session(self, session_id: str):
        """End a session"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.is_active = False
                self.logger.info(f"Ended session: {session_id}")
    
    def _cleanup_expired_sessions(self, force: bool = False):
        """Clean up expired sessions"""
        with self._lock:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                # Check if session is too old
                age_hours = (current_time - session.created_at).total_seconds() / 3600
                if age_hours > session.max_session_age_hours:
                    expired_sessions.append(session_id)
                    continue
                
                # Check if session has been inactive for too long (1 hour)
                inactive_hours = (current_time - session.last_activity).total_seconds() / 3600
                if inactive_hours > 1:
                    expired_sessions.append(session_id)
                    continue
                
                # Check if session is marked as inactive
                if not session.is_active:
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self.sessions[session_id]
                self.logger.info(f"Cleaned up expired session: {session_id}")
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        with self._lock:
            active_sessions = sum(1 for s in self.sessions.values() if s.is_active)
            total_sessions = len(self.sessions)
            
            return {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'max_sessions': self.max_sessions,
                'cleanup_interval': self.cleanup_interval
            }
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user"""
        with self._lock:
            return [
                session_id for session_id, session in self.sessions.items()
                if session.user_id == user_id and session.is_active
            ]
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if a session exists and is active"""
        session = self.get_session(session_id)
        return session is not None and session.is_active

# Global session manager instance
session_manager = SessionManager() 