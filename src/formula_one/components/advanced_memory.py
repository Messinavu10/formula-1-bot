import chromadb
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict
from collections import defaultdict

from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.config_entity import DatabaseConfig

@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    id: str
    timestamp: str
    user_query: str
    assistant_response: str
    entities: Dict[str, List[str]]  # drivers, teams, races, query_types
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ConversationSummary:
    """Rolling summary of conversation"""
    current_race: Optional[str] = None
    current_session: Optional[str] = None
    active_drivers: List[str] = None
    active_teams: List[str] = None
    analysis_type: Optional[str] = None
    last_update: str = None
    
    def __post_init__(self):
        if self.active_drivers is None:
            self.active_drivers = []
        if self.active_teams is None:
            self.active_teams = []
        if self.last_update is None:
            self.last_update = datetime.now().isoformat()

class VectorMemoryManager(BaseComponent):
    """Advanced vector database memory system for F1 conversations"""
    
    def __init__(self, config, db_config: DatabaseConfig):
        super().__init__(config, db_config)
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./artifacts/memory")
        self.collection = self.chroma_client.get_or_create_collection(
            name="f1_conversations",
            metadata={"description": "F1 conversation memory"}
        )
        
        # Entity memory (fast access)
        self.entity_memory = {
            "drivers": set(),
            "teams": set(),
            "races": set(),
            "query_types": set()
        }
        
        # Conversation summary
        self.conversation_summary = ConversationSummary()
        
        # Context cache for recent exchanges
        self.context_cache = []
        self.max_cache_size = 10
        
        # Load existing memory
        self._load_existing_memory()
        
        self.logger.info("🧠 Vector Memory Manager initialized successfully")
    
    def _load_existing_memory(self):
        """Load existing memory from ChromaDB"""
        try:
            # Get all existing entries
            results = self.collection.get()
            if results and results['ids']:
                self.logger.info(f"📚 Loaded {len(results['ids'])} existing memory entries")
                
                # Rebuild entity memory
                for metadata in results['metadatas']:
                    if metadata and 'entities' in metadata:
                        entities = json.loads(metadata['entities'])
                        for entity_type, values in entities.items():
                            if entity_type in self.entity_memory:
                                self.entity_memory[entity_type].update(values)
        except Exception as e:
            self.logger.warning(f"Could not load existing memory: {e}")
    
    def store_exchange(self, user_query: str, assistant_response: str, query_analysis: Dict[str, Any]):
        """Store a conversation exchange in memory"""
        try:
            # Extract entities from query analysis
            entities = self._extract_entities_from_analysis(query_analysis)
            
            # Generate embedding for the exchange
            exchange_text = f"User: {user_query}\nAssistant: {assistant_response}"
            embedding = self.embedding_model.encode(exchange_text).tolist()
            
            # Create memory entry
            memory_entry = MemoryEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                user_query=user_query,
                assistant_response=assistant_response,
                entities=entities,
                embedding=embedding,
                metadata={
                    "query_type": query_analysis.get("query_type"),
                    "meeting_info": query_analysis.get("meeting_info"),
                    "session_type": query_analysis.get("session_type")
                }
            )
            
            # Store in ChromaDB
            self.collection.add(
                ids=[memory_entry.id],
                embeddings=[embedding],
                documents=[exchange_text],
                metadatas=[{
                    "entities": json.dumps(entities),
                    "query_type": query_analysis.get("query_type"),
                    "meeting_info": json.dumps(query_analysis.get("meeting_info", {})),
                    "session_type": query_analysis.get("session_type"),
                    "timestamp": memory_entry.timestamp
                }]
            )
            
            # Update entity memory
            self._update_entity_memory(entities)
            
            # Update conversation summary
            self._update_conversation_summary(query_analysis)
            
            # Update context cache
            self._update_context_cache(memory_entry)
            
            self.logger.info(f"�� Stored exchange in memory: {memory_entry.id}")
            self.logger.info(f"🔍 Entities: {entities}")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing exchange: {e}")
    
    def retrieve_relevant_context(self, user_query: str, query_analysis: Dict[str, Any], max_results: int = 5) -> Dict[str, Any]:
        """Retrieve relevant context for the current query"""
        try:
            # Generate embedding for current query
            query_embedding = self.embedding_model.encode(user_query).tolist()
            
            # Search for similar conversations
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=["metadatas", "documents"]
            )
            
            # Extract relevant context
            relevant_exchanges = []
            entity_matches = []
            
            if search_results['ids'] and search_results['ids'][0]:
                for i, doc_id in enumerate(search_results['ids'][0]):
                    metadata = search_results['metadatas'][0][i]
                    document = search_results['documents'][0][i]
                    
                    # Parse entities from metadata
                    entities = json.loads(metadata.get('entities', '{}'))
                    
                    # Check for entity matches
                    current_entities = self._extract_entities_from_analysis(query_analysis)
                    entity_overlap = self._calculate_entity_overlap(current_entities, entities)
                    
                    if entity_overlap > 0:
                        entity_matches.append({
                            "exchange": document,
                            "entities": entities,
                            "overlap_score": entity_overlap,
                            "query_type": metadata.get('query_type'),
                            "meeting_info": json.loads(metadata.get('meeting_info', '{}')),
                            "session_type": metadata.get('session_type')
                        })
                    
                    relevant_exchanges.append({
                        "exchange": document,
                        "entities": entities,
                        "query_type": metadata.get('query_type'),
                        "meeting_info": json.loads(metadata.get('meeting_info', '{}')),
                        "session_type": metadata.get('session_type')
                    })
            
            # Sort by relevance (entity overlap first, then semantic similarity)
            entity_matches.sort(key=lambda x: x['overlap_score'], reverse=True)
            
            # Combine context
            context = {
                "relevant_exchanges": relevant_exchanges[:3],
                "entity_matches": entity_matches[:3],
                "conversation_summary": asdict(self.conversation_summary),
                "entity_memory": {k: list(v) for k, v in self.entity_memory.items()},
                "context_cache": self.context_cache[-3:]  # Last 3 exchanges
            }
            
            self.logger.info(f"�� Retrieved context: {len(relevant_exchanges)} exchanges, {len(entity_matches)} entity matches")
            
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving context: {e}")
            return {
                "relevant_exchanges": [],
                "entity_matches": [],
                "conversation_summary": asdict(self.conversation_summary),
                "entity_memory": {k: list(v) for k, v in self.entity_memory.items()},
                "context_cache": self.context_cache[-3:]
            }
    
    def _extract_entities_from_analysis(self, query_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract entities from query analysis"""
        entities = {
            "drivers": query_analysis.get("drivers", []),
            "teams": query_analysis.get("teams", []),
            "races": [],
            "query_types": [query_analysis.get("query_type", "unknown")]
        }
        
        # Extract race from meeting info
        meeting_info = query_analysis.get("meeting_info", {})
        if meeting_info.get("name"):
            entities["races"].append(meeting_info["name"])
        
        return entities
    
    def _update_entity_memory(self, entities: Dict[str, List[str]]):
        """Update entity memory with new entities"""
        for entity_type, values in entities.items():
            if entity_type in self.entity_memory:
                self.entity_memory[entity_type].update(values)
    
    def _update_conversation_summary(self, query_analysis: Dict[str, Any]):
        """Update conversation summary"""
        meeting_info = query_analysis.get("meeting_info", {})
        if meeting_info.get("name"):
            self.conversation_summary.current_race = meeting_info["name"]
        
        session_type = query_analysis.get("session_type")
        if session_type:
            self.conversation_summary.current_session = session_type
        
        drivers = query_analysis.get("drivers", [])
        if drivers:
            self.conversation_summary.active_drivers.extend(drivers)
            # Keep only unique drivers
            self.conversation_summary.active_drivers = list(set(self.conversation_summary.active_drivers))
        
        teams = query_analysis.get("teams", [])
        if teams:
            self.conversation_summary.active_teams.extend(teams)
            # Keep only unique teams
            self.conversation_summary.active_teams = list(set(self.conversation_summary.active_teams))
        
        query_type = query_analysis.get("query_type")
        if query_type:
            self.conversation_summary.analysis_type = query_type
        
        self.conversation_summary.last_update = datetime.now().isoformat()
    
    def _update_context_cache(self, memory_entry: MemoryEntry):
        """Update context cache with new exchange"""
        self.context_cache.append({
            "timestamp": memory_entry.timestamp,
            "user_query": memory_entry.user_query,
            "assistant_response": memory_entry.assistant_response,
            "entities": memory_entry.entities
        })
        
        # Keep cache size manageable
        if len(self.context_cache) > self.max_cache_size:
            self.context_cache.pop(0)
    
    def _calculate_entity_overlap(self, current_entities: Dict[str, List[str]], stored_entities: Dict[str, List[str]]) -> float:
        """Calculate overlap between current and stored entities"""
        total_overlap = 0
        total_entities = 0
        
        for entity_type in ["drivers", "teams", "races"]:
            current_set = set(current_entities.get(entity_type, []))
            stored_set = set(stored_entities.get(entity_type, []))
            
            if current_set and stored_set:
                overlap = len(current_set.intersection(stored_set))
                total_entities += len(current_set.union(stored_set))
                total_overlap += overlap
        
        if total_entities == 0:
            return 0.0
        
        return total_overlap / total_entities
    
    def detect_topic_change(self, query_analysis: Dict[str, Any]) -> bool:
        """Detect if there's a significant topic change"""
        meeting_info = query_analysis.get("meeting_info", {})
        current_race = meeting_info.get("name")
        
        # Check if race has changed
        if current_race and self.conversation_summary.current_race:
            if current_race != self.conversation_summary.current_race:
                self.logger.info(f"🔄 Topic change detected: {self.conversation_summary.current_race} → {current_race}")
                return True
        
        # Check if analysis type has changed significantly
        current_query_type = query_analysis.get("query_type")
        if current_query_type and self.conversation_summary.analysis_type:
            if current_query_type != self.conversation_summary.analysis_type:
                # Some changes are more significant than others
                significant_changes = [
                    ("comparison", "driver_performance"),
                    ("driver_performance", "team_performance"),
                    ("race_results", "qualifying_results")
                ]
                
                for old_type, new_type in significant_changes:
                    if (self.conversation_summary.analysis_type == old_type and current_query_type == new_type) or \
                       (self.conversation_summary.analysis_type == new_type and current_query_type == old_type):
                        self.logger.info(f"🔄 Analysis type change: {self.conversation_summary.analysis_type} → {current_query_type}")
                        return True
        
        return False
    
    def clear_context_for_new_topic(self):
        """Clear context when starting a new topic"""
        self.conversation_summary = ConversationSummary()
        self.context_cache.clear()
        self.logger.info("🧹 Cleared context for new topic")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            results = self.collection.get()
            total_entries = len(results['ids']) if results['ids'] else 0
            
            return {
                "total_entries": total_entries,
                "entity_memory": {k: len(v) for k, v in self.entity_memory.items()},
                "conversation_summary": asdict(self.conversation_summary),
                "context_cache_size": len(self.context_cache)
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting memory stats: {e}")
            return {"error": str(e)}