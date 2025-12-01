"""
Memory Service V2.1 (Restored Logic + Multi-User Support)
Connects modular services and restores core search/decay algorithms.
"""
import time
import json
import logging
import asyncio
import re
from typing import Optional, List, Dict, Any, Set
from collections import Counter

# Import Modular Services (Created by CLI)
import flight_recorder_service
import qdrant_client_wrapper
import embedding_helper

# Re-export keys for compatibility
read_rec = flight_recorder_service.read_rec
log_turn_async = flight_recorder_service.log_turn_async
update_last_turn_response_async = flight_recorder_service.update_last_turn_response_async
FLIGHT_DIR = flight_recorder_service.FLIGHT_DIR

# Import Qdrant models needed for logic
from qdrant_client.http.models import (
    Filter, FieldCondition, MatchValue, IsNullCondition, PayloadField,
    Prefetch, Fusion, FusionQuery, NamedVector, SparseVector, PointStruct
)

# --- GLOBAL CONFIG ---
COLLECTION = "decay_memory_mcp"
ENABLE_HYBRID_SEARCH = True
SPARSE_VECTOR_NAME = "keywords"

# Tuning Constants (From our tests)
DECAY_FLOOR = 0.3
PROACTIVE_CONFIG = {
    "enabled": True,
    "trigger_threshold": 0.60,
    "max_injections": 3,
    "fast_path_threshold": 0.50,
    "injection_threshold": 0.15 # Tuned low for recall
}

# Stop words for sparse vector generation
STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'it', 'that', 'this'}

logger = logging.getLogger("DecayMemory")

class MemorySystem:
    def __init__(self):
        # 1. Initialize DB via Wrapper
        self.client = qdrant_client_wrapper.get_qdrant_client()
        qdrant_client_wrapper.init_db(self.client)
        logger.info("Memory System Initialized (Modular V2.1)")

    # --- HELPER: Sparse Vectors ---
    def _generate_sparse_vector(self, text: str) -> Dict[int, float]:
        """Generate sparse vector for hybrid search."""
        import hashlib
        words = re.findall(r'\b[a-z0-9]+\b', text.lower())
        words = [w for w in words if w not in STOP_WORDS]
        if not words: return {0: 0.001}
        
        counts = Counter(words)
        max_count = max(counts.values())
        vec = {}
        for w, c in counts.items():
            # Hash to int32 index
            idx = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % 1000000
            vec[idx] = c / max_count
        return vec

    # --- CORE: Project Filtering (Updated for Multi-User) ---
    def get_project_filter(self, proj: Optional[str], include_global: bool = True) -> Optional[Filter]:
        """Builds filter for Project + Global scope."""
        if proj in [None, "", "None", "global"]:
            proj = ""
            
        global_conditions = [
            FieldCondition(key="project_name", match=MatchValue(value="")),
            IsNullCondition(is_null=PayloadField(key="project_name")),
            FieldCondition(key="project_name", match=MatchValue(value="global"))
        ]
        
        if not proj:
            return Filter(should=global_conditions)
            
        if include_global:
            conditions = [FieldCondition(key="project_name", match=MatchValue(value=proj))]
            conditions.extend(global_conditions)
            return Filter(should=conditions)
        
        return Filter(must=[FieldCondition(key="project_name", match=MatchValue(value=proj))])

    # --- CORE: Search Logic (Restored) ---
    def hybrid_search(self, query: str, proj: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """Performs Hybrid Search (Dense + Sparse)."""
        # Use helper for embedding
        dense_vec = embedding_helper.embed(query)
        if not dense_vec: return []
        
        proj_filter = self.get_project_filter(proj, include_global=True)
        results = []
        
        try:
            if ENABLE_HYBRID_SEARCH:
                sparse_vec = self._generate_sparse_vector(query)
                prefetch = [
                    Prefetch(
                        query=dense_vec, using="dense", filter=proj_filter, limit=limit*2
                    ),
                    Prefetch(
                        query=SparseVector(indices=list(sparse_vec.keys()), values=list(sparse_vec.values())),
                        using=SPARSE_VECTOR_NAME, filter=proj_filter, limit=limit*2
                    )
                ]
                search_result = self.client.query_points(
                    COLLECTION, prefetch=prefetch, query=FusionQuery(fusion=Fusion.RRF),
                    limit=limit, with_payload=True
                )
                results = search_result.points
            else:
                results = self.client.search(
                    COLLECTION, query_vector=dense_vec, query_filter=proj_filter,
                    limit=limit, with_payload=True
                )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        # Process Results (Decay Calculation)
        processed = []
        for hit in results:
            age = time.time() - hit.payload.get('timestamp', 0)
            # Default half-life: 1 week (604800s)
            hl = 604800 
            decay = max(0.5 ** (age / hl), DECAY_FLOOR)
            boost = hit.payload.get('boost_factor', 1.0)
            final_score = hit.score * decay * boost
            
            processed.append(hit.payload | {'id': hit.id, 'search_score': final_score})
            
        processed.sort(key=lambda x: x['search_score'], reverse=True)
        return processed[:limit]

    # --- CORE: Memory Storage ---
    def add_memory(self, user_id: str, content: str, mtype: str = "personal", proj: str = "global") -> str:
        """Saves a new memory (Restored Logic)."""
        import uuid
        
        dense_vec = embedding_helper.embed(content)
        if not dense_vec: return "Error: Embedding failed"
        
        pid = str(uuid.uuid4())
        payload = {
            "content": content,
            "type": mtype,
            "project_name": proj,
            "user_id": user_id, # Added User ID support
            "timestamp": time.time(),
            "score": 1.0,
            "boost_factor": 1.0
        }
        
        if ENABLE_HYBRID_SEARCH:
            sparse_vec = self._generate_sparse_vector(content)
            point = PointStruct(
                id=pid,
                vector={
                    "dense": dense_vec,
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=list(sparse_vec.keys()),
                        values=list(sparse_vec.values())
                    )
                },
                payload=payload
            )
        else:
            point = PointStruct(id=pid, vector=dense_vec, payload=payload)
            
        self.client.upsert(COLLECTION, points=[point])
        return "Memory saved."

    # --- API Wrapper ---
    def proactive_retrieval(self, user_id: str, user_message: str, project_name: str) -> Optional[str]:
        """
        Public API for the Orchestrator.
        Parses the message -> Searches DB -> Formats Context.
        """
        # Logic: 
        # 1. Trigger Detection (Simplified for now: Search everything)
        # 2. Hybrid Search
        # 3. Format
        
        results = self.hybrid_search(user_message, proj=project_name)
        
        # Filter by threshold
        valid_memories = [
            m for m in results 
            if m['search_score'] >= PROACTIVE_CONFIG['injection_threshold']
        ]
        
        if not valid_memories: return None
        
        context_str = ""
        for i, mem in enumerate(valid_memories, 1):
            context_str += f"{i}. {mem.get('content')} (Relevance: {mem['search_score']:.2f})\n"
            
        return context_str

    # --- Session Updates (Bridge to Flight Recorder) ---
    def update_session(self, user_id: str, msg: str, proj: str):
        # In a real multi-user system, we'd store session state in Qdrant per user.
        # For now, we just acknowledge it.
        pass