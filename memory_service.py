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

# Feedback System Constants (from v1.0 dashboard_config.json)
BOOST_INCREMENT = 1.5      # Multiply boost by this each time
MAX_BOOST = 4.0            # Maximum boost multiplier
DEPRECATION_MULTIPLIER = 0.1  # Decay 10x faster when deprecated

# Conflict Resolution Constants (from v1.0)
CONFLICT_TYPES = {'personal', 'preference', 'goal', 'project'}  # Types to check for conflicts
CONFLICT_SIMILARITY_THRESHOLD = 0.85  # Similarity score to consider as potential conflict
CONFLICT_CONFIDENCE_THRESHOLD = 0.8   # LLM confidence required to auto-resolve

PROACTIVE_CONFIG = {
    "enabled": True,
    "trigger_threshold": 0.60,
    "max_injections": 3,
    "fast_path_threshold": 0.50,
    "injection_threshold": 0.15 # Tuned low for recall
}

# Stop words for sparse vector generation
STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'it', 'that', 'this'}

# Session Management Constants (from v1.0)
EMBED_DIM = 1536  # OpenAI embedding dimension
MAX_RAW_SUMMARY_LENGTH = 500  # Max chars for raw summary fallback

logger = logging.getLogger("DecayMemory")

# =============================================================================
# HELPER FUNCTIONS (Restored from v1.0)
# =============================================================================

import hashlib

def get_int_id(seed: str, offset: int) -> int:
    """
    Generate an integer ID from seed with offset.
    Uses 8 hex characters (32 bits) with modulo for unique values per offset range.
    """
    return offset + (int(hashlib.md5(seed.encode()).hexdigest()[:8], 16) % 89999989)


def create_empty_point(point_id: int, payload: Dict[str, Any]) -> PointStruct:
    """Create a PointStruct with zero vectors (for session state, etc.)."""
    if ENABLE_HYBRID_SEARCH:
        return PointStruct(
            id=point_id,
            vector={
                "dense": [0.0] * EMBED_DIM,
                SPARSE_VECTOR_NAME: SparseVector(indices=[], values=[])
            },
            payload=payload
        )
    else:
        return PointStruct(id=point_id, vector=[0.0] * EMBED_DIM, payload=payload)


def condense_text(text: str, aggressive: bool = False) -> str:
    """Remove conversational filler while preserving semantic content."""
    if not text:
        return text
    
    filler_patterns = [
        r'\b(good morning|good afternoon|good evening|hello|hi there|hey)\b',
        r'\b(please|thank you|thanks|thank you very much)\b',
        r'\b(I think|I believe|I feel like|it seems like|I\'m wondering)\b',
        r'\b(kind of|sort of|basically|essentially|actually)\b',
        r'\b(you know|I mean|like|well|so|um|uh)\b',
        r'\b(just|really|very|quite|pretty)\b',
        r'\b(a bit|a little|somewhat)\b',
    ]
    
    condensed = text
    for pattern in filler_patterns:
        condensed = re.sub(pattern, '', condensed, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    condensed = re.sub(r'\s+', ' ', condensed).strip()
    return condensed


def format_condensed_turn(user_msg: str, ai_summary: str, timestamp: float) -> str:
    """Format a conversation turn in condensed format."""
    now = time.time()
    hours_ago = (now - timestamp) / 3600
    
    if hours_ago < 1:
        time_str = "just now"
    elif hours_ago < 24:
        time_str = f"{hours_ago:.1f}h ago"
    else:
        time_str = f"{hours_ago/24:.1f}d ago"
    
    c_user = condense_text(user_msg)
    c_ai = condense_text(ai_summary)
    return f"[{time_str}] {c_user}\n→ {c_ai}"


def format_recent_turns(turns: List[Dict[str, Any]]) -> str:
    """Format turns in full detail (not condensed)."""
    formatted = []
    for t in turns:
        ts = t.get('ts', 0)
        user = t.get('turn', {}).get('user', '')
        ai = t.get('turn', {}).get('ai', '(Pending)')
        formatted.append(f"USER: {user}\nAI: {ai}")
    return "\n\n---\n\n".join(formatted)

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

    # =========================================================================
    # MEMORY FEEDBACK SYSTEM (Restored from v1.0)
    # =========================================================================
    
    def boost_memory(self, memory_id: str, reason: str = "") -> str:
        """
        Increase memory importance by multiplying boost_factor.
        Each boost multiplies by BOOST_INCREMENT (1.5x), up to MAX_BOOST (4x).
        """
        try:
            res = self.client.retrieve(COLLECTION, ids=[memory_id], with_payload=True)
            if not res:
                return "❌ Memory not found"
            
            mem = res[0]
            curr = mem.payload.get('boost_factor', 1.0)
            new_boost = min(curr * BOOST_INCREMENT, MAX_BOOST)
            
            # Track boost history
            hist = mem.payload.get('boost_history', [])
            hist.append({
                "ts": time.time(), 
                "reason": reason, 
                "old": curr, 
                "new": new_boost
            })
            
            self.client.set_payload(
                COLLECTION,
                {"boost_factor": new_boost, "boost_history": hist},
                points=[memory_id]
            )
            return f"✅ Boosted {curr:.1f}x → {new_boost:.1f}x"
        except Exception as e:
            logger.error(f"boost_memory failed: {e}")
            return f"❌ Error: {e}"
    
    def deprecate_memory(self, memory_id: str, reason: str = "") -> str:
        """
        Mark memory as deprecated - it will decay 10x faster.
        Deprecated memories still exist but fade quickly.
        """
        try:
            self.client.set_payload(
                COLLECTION,
                {
                    "is_deprecated": True,
                    "deprecation_factor": DEPRECATION_MULTIPLIER,
                    "deprecation_reason": reason
                },
                points=[memory_id]
            )
            return f"✅ Memory deprecated: {reason or 'No reason given'}"
        except Exception as e:
            logger.error(f"deprecate_memory failed: {e}")
            return f"❌ Error: {e}"
    
    def correct_memory(self, memory_id: str, new_content: str) -> str:
        """
        Update memory content with new text.
        Regenerates embedding and preserves correction history.
        """
        try:
            res = self.client.retrieve(COLLECTION, ids=[memory_id], with_payload=True)
            if not res:
                return "❌ Memory not found"
            
            mem = res[0]
            old_content = mem.payload.get('content', '')
            
            # Track correction history
            hist = mem.payload.get('correction_history', [])
            hist.append({
                "ts": time.time(),
                "old": old_content,
                "new": new_content
            })
            
            # Generate new embedding
            new_vec = embedding_helper.embed(new_content)
            if new_vec is None:
                return "❌ Error: Failed to generate embedding"
            
            # Update payload
            updated_payload = mem.payload.copy()
            updated_payload.update({
                "content": new_content,
                "timestamp": time.time(),
                "correction_history": hist
            })
            
            # Create new point with updated vector and payload
            if ENABLE_HYBRID_SEARCH:
                sparse_vec = self._generate_sparse_vector(new_content)
                point = PointStruct(
                    id=memory_id,
                    vector={
                        "dense": new_vec,
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=list(sparse_vec.keys()),
                            values=list(sparse_vec.values())
                        )
                    },
                    payload=updated_payload
                )
            else:
                point = PointStruct(id=memory_id, vector=new_vec, payload=updated_payload)
            
            self.client.upsert(COLLECTION, points=[point])
            return f"✅ Memory corrected: '{old_content[:50]}...' → '{new_content[:50]}...'"
        except Exception as e:
            logger.error(f"correct_memory failed: {e}")
            return f"❌ Error: {e}"
    
    def delete_memory(self, memory_id: str, reason: str = "") -> str:
        """
        Soft-delete a memory by archiving it.
        The memory is preserved but hidden from searches.
        """
        try:
            self.client.set_payload(
                COLLECTION,
                {
                    "type": "archived_deleted",
                    "del_reason": reason,
                    "deleted_at": time.time()
                },
                points=[memory_id]
            )
            return f"✅ Memory archived: {reason or 'No reason given'}"
        except Exception as e:
            logger.error(f"delete_memory failed: {e}")
            return f"❌ Error: {e}"
    
    def set_memory_project(self, memory_id: str, project_name: str) -> str:
        """
        Move a memory to a different project.
        """
        try:
            res = self.client.retrieve(COLLECTION, ids=[memory_id], with_payload=True)
            if not res:
                return "❌ Memory not found"
            
            old_project = res[0].payload.get('project_name', 'global')
            self.client.set_payload(
                COLLECTION,
                {"project_name": project_name},
                points=[memory_id]
            )
            return f"✅ Moved: '{old_project}' → '{project_name or 'global'}'"
        except Exception as e:
            logger.error(f"set_memory_project failed: {e}")
            return f"❌ Error: {e}"
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single memory by ID.
        Useful for inspecting memory details.
        """
        try:
            res = self.client.retrieve(COLLECTION, ids=[memory_id], with_payload=True)
            if not res:
                return None
            return res[0].payload | {"id": res[0].id}
        except Exception as e:
            logger.error(f"get_memory failed: {e}")
            return None

    # =========================================================================
    # CONFLICT RESOLUTION SYSTEM (Restored from v1.0)
    # =========================================================================
    
    def check_conflict(self, content: str, mtype: str, proj: str) -> Optional[Dict[str, Any]]:
        """
        Check for existing memories that might conflict with new content.
        Returns the conflicting memory if similarity > threshold, else None.
        """
        if mtype not in CONFLICT_TYPES:
            return None
        
        vec = embedding_helper.embed(content)
        if vec is None:
            logger.warning("Failed to generate embedding for conflict check")
            return None
        
        # Build filter: must match type AND (project OR global)
        type_condition = FieldCondition(key="type", match=MatchValue(value=mtype))
        
        if proj and proj not in ["", "global", "*"]:
            # Project-specific: include project memories and global
            combined_filter = Filter(
                must=[type_condition],
                should=[
                    FieldCondition(key="project_name", match=MatchValue(value=proj)),
                    FieldCondition(key="project_name", match=MatchValue(value="")),
                    FieldCondition(key="project_name", match=MatchValue(value="global")),
                    IsNullCondition(is_null=PayloadField(key="project_name"))
                ]
            )
        else:
            # Global only
            combined_filter = Filter(
                must=[type_condition],
                should=[
                    FieldCondition(key="project_name", match=MatchValue(value="")),
                    FieldCondition(key="project_name", match=MatchValue(value="global")),
                    IsNullCondition(is_null=PayloadField(key="project_name"))
                ]
            )
        
        try:
            # Use query_points with dense vector (same approach as hybrid_search)
            search_result = self.client.query_points(
                COLLECTION,
                query=vec,
                using="dense",
                query_filter=combined_filter,
                limit=5,
                with_payload=True
            )
            
            for h in search_result.points:
                if h.score >= CONFLICT_SIMILARITY_THRESHOLD:
                    logger.info(f"Potential conflict found: {h.payload.get('content', '')[:50]}... (score: {h.score:.2f})")
                    return h.payload | {'id': h.id, 'similarity': h.score}
            return None
        except Exception as e:
            logger.error(f"check_conflict failed: {e}")
            return None
    
    def resolve_conflict(self, new_content: str, old_memory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Use Gemini LLM to determine relationship between conflicting memories.
        Returns: {"rel": "supersede|update|complement|unrelated", "merged": "...", "conf": 0.9}
        """
        try:
            import google.generativeai as genai
            import os
            
            # Configure Gemini if not already
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not set, cannot resolve conflict via LLM")
                return None
            
            genai.configure(api_key=api_key)
            
            prompt = (
                f'Compare these two pieces of information about the same topic:\n\n'
                f'EXISTING: "{old_memory.get("content")}"\n'
                f'NEW: "{new_content}"\n\n'
                f'Determine their relationship:\n'
                f'1. SUPERSEDE - New completely replaces old (old is outdated)\n'
                f'2. UPDATE - New modifies/corrects old (merge them)\n'
                f'3. COMPLEMENT - Both are valid, can coexist\n'
                f'4. UNRELATED - Not actually about the same thing\n\n'
                f'Return JSON: {{ "rel": "supersede|update|complement|unrelated", "merged": "merged text if UPDATE", "conf": confidence_0_to_1, "reason": "brief explanation" }}'
            )
            
            model = genai.GenerativeModel(
                'gemini-2.0-flash',
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content(prompt)
            result = json.loads(response.text)
            logger.info(f"Conflict resolution: {result.get('rel')} (conf: {result.get('conf', 0):.2f})")
            return result
        except Exception as e:
            logger.error(f"resolve_conflict failed: {e}")
            return None
    
    def replace_memory(self, old_id: str, new_content: str, old_mem: Dict[str, Any]) -> Optional[str]:
        """
        Replace an old memory with new content.
        Archives the old memory and creates new one with same metadata.
        """
        import uuid
        
        dense_vec = embedding_helper.embed(new_content)
        if dense_vec is None:
            logger.warning("Failed to generate embedding for replacement")
            return None
        
        try:
            # Archive old memory (soft delete, preserves history)
            self.client.set_payload(
                COLLECTION,
                {
                    "type": "archived_superseded",
                    "superseded_at": time.time(),
                    "superseded_reason": "conflict_resolution"
                },
                points=[old_id]
            )
            
            # Create new memory with same metadata
            new_payload = {
                "content": new_content,
                "type": old_mem.get('type', 'personal'),
                "project_name": old_mem.get('project_name', 'global'),
                "user_id": old_mem.get('user_id', 'default_user'),
                "timestamp": time.time(),
                "score": 1.0,
                "boost_factor": old_mem.get('boost_factor', 1.0),  # Preserve boost
                "supersedes": old_id  # Link to old memory
            }
            
            pid = str(uuid.uuid4())
            
            if ENABLE_HYBRID_SEARCH:
                sparse_vec = self._generate_sparse_vector(new_content)
                point = PointStruct(
                    id=pid,
                    vector={
                        "dense": dense_vec,
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=list(sparse_vec.keys()),
                            values=list(sparse_vec.values())
                        )
                    },
                    payload=new_payload
                )
            else:
                point = PointStruct(id=pid, vector=dense_vec, payload=new_payload)
            
            self.client.upsert(COLLECTION, points=[point])
            logger.info(f"Replaced memory {old_id} with {pid}")
            return pid
        except Exception as e:
            logger.error(f"replace_memory failed: {e}")
            return None
    
    def add_memory_with_conflict_check(
        self, 
        user_id: str, 
        content: str, 
        mtype: str = "personal", 
        proj: str = "global",
        auto_resolve: bool = True
    ) -> Dict[str, Any]:
        """
        Add a memory with automatic conflict detection and resolution.
        
        Returns dict with:
        - status: "added" | "superseded" | "merged" | "conflict_detected"
        - memory_id: the ID of the new/updated memory
        - details: additional info about what happened
        """
        import uuid
        
        # Check for conflicts
        conflict = self.check_conflict(content, mtype, proj)
        
        if conflict and auto_resolve:
            resolution = self.resolve_conflict(content, conflict)
            
            if resolution and resolution.get('conf', 0) > CONFLICT_CONFIDENCE_THRESHOLD:
                rel = resolution.get('rel', '').lower()
                
                if rel == 'supersede':
                    new_id = self.replace_memory(conflict['id'], content, conflict)
                    return {
                        "status": "superseded",
                        "memory_id": new_id,
                        "old_id": conflict['id'],
                        "details": f"Superseded old memory: {conflict.get('content', '')[:50]}..."
                    }
                
                elif rel in ['update', 'complement']:
                    merged_content = resolution.get('merged', content)
                    new_id = self.replace_memory(conflict['id'], merged_content, conflict)
                    return {
                        "status": "merged",
                        "memory_id": new_id,
                        "old_id": conflict['id'],
                        "details": f"Merged: {merged_content[:50]}..."
                    }
                
                # 'unrelated' falls through to normal add
        
        elif conflict and not auto_resolve:
            # Return conflict info without resolving
            return {
                "status": "conflict_detected",
                "memory_id": None,
                "conflict": conflict,
                "details": f"Potential conflict with: {conflict.get('content', '')[:50]}..."
            }
        
        # No conflict or unrelated - normal add
        dense_vec = embedding_helper.embed(content)
        if not dense_vec:
            return {"status": "error", "memory_id": None, "details": "Embedding failed"}
        
        pid = str(uuid.uuid4())
        payload = {
            "content": content,
            "type": mtype,
            "project_name": proj,
            "user_id": user_id,
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
        return {"status": "added", "memory_id": pid, "details": "Memory saved"}

    # =========================================================================
    # SESSION MANAGEMENT (Restored from v1.0)
    # =========================================================================
    
    def reset_session(self, proj: Optional[str], init_only: bool = False) -> None:
        """Reset or initialize session state."""
        import uuid as uuid_module
        
        sid = get_int_id(f"session_{proj}", 10)
        if init_only:
            try:
                existing = self.client.retrieve(COLLECTION, ids=[sid])
                if existing:
                    return
            except Exception:
                pass
        
        payload = {
            "type": "session_state",
            "turn_count": 0,
            "tokens_used": 0,
            "session_start": time.time(),
            "last_update": time.time(),
            "session_id": str(uuid_module.uuid4()),
            "project_name": proj or ""
        }
        
        point = create_empty_point(sid, payload)
        self.client.upsert(COLLECTION, points=[point])
    
    def get_session_state(self, proj: Optional[str]) -> Dict[str, Any]:
        """Get current session state for a project."""
        sid = get_int_id(f"session_{proj}", 10)
        try:
            res = self.client.retrieve(COLLECTION, ids=[sid])
            if res:
                return res[0].payload
        except Exception:
            pass
        
        # Initialize if not found
        self.reset_session(proj)
        res = self.client.retrieve(COLLECTION, ids=[sid])
        return res[0].payload if res else {}
    
    def update_session_turn(self, proj: Optional[str]) -> Dict[str, Any]:
        """Update session turn count and metadata."""
        sid = get_int_id(f"session_{proj}", 10)
        try:
            res = self.client.retrieve(COLLECTION, ids=[sid])
            if not res:
                self.reset_session(proj)
                res = self.client.retrieve(COLLECTION, ids=[sid])
            
            st = res[0].payload
            st['turn_count'] = st.get('turn_count', 0) + 1
            st['last_update'] = time.time()
            
            point = create_empty_point(sid, st)
            self.client.upsert(COLLECTION, points=[point])
            return {"state": st}
        except Exception as e:
            logger.error(f"update_session_turn failed: {e}")
            return {"error": str(e)}
    
    async def finalize_session_async(self, summary: str, proj: str) -> str:
        """
        Finalize session with LLM extraction and handover creation.
        
        1. Extract facts from summary via Gemini
        2. Store as memories
        3. Create handover document for next session
        4. Clean up flight recorder
        5. Reset session state
        """
        import os
        
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            gemini_configured = bool(api_key)
            if gemini_configured:
                genai.configure(api_key=api_key)
        except ImportError:
            gemini_configured = False
        
        if not gemini_configured:
            logger.warning("Gemini not configured, skipping LLM summarization")
            # Cleanup old turns
            await flight_recorder_service.cleanup_old_turns_async("default_user", proj)
            self.reset_session(proj)
            return "✅ Session Finalized (without LLM summary)."
        
        # Try to extract structured facts
        prompt = f"Extract facts JSON list. Types: personal,project,topic.\n{summary}"
        try:
            result = genai.GenerativeModel(
                'gemini-2.0-flash',
                generation_config={"response_mime_type": "application/json"}
            ).generate_content(prompt)
            mems = json.loads(result.text)
            for m in mems:
                self.add_memory("default_user", m.get('content', ''), m.get('type', 'topic'), proj)
        except Exception as e:
            logger.warning(f"Finalize extraction failed: {e}. Storing raw summary.")
            self.add_memory(
                "default_user",
                f"Session Summary (Raw): {summary[:MAX_RAW_SUMMARY_LENGTH]}",
                "dialog",
                proj
            )
        
        # Create handover document
        try:
            prompt = (
                f"Summarize for next session. Project: {proj}. "
                f"JSON: summary, last_topic, unresolved.\n{summary}"
            )
            result = genai.GenerativeModel(
                'gemini-2.0-flash',
                generation_config={"response_mime_type": "application/json"}
            ).generate_content(prompt)
            ho = json.loads(result.text)
            hid = get_int_id(f"handover_{proj}", 100000000)
            vec = embedding_helper.embed(ho.get('summary', summary[:200]))
            
            if vec is None:
                logger.warning("Failed to generate embedding for handover")
            else:
                payload = {
                    "content": ho.get('summary', ''),
                    "type": "context_handover",
                    "project_name": proj or "",
                    "timestamp": time.time(),
                    **ho
                }
                
                if ENABLE_HYBRID_SEARCH:
                    sparse_vec = self._generate_sparse_vector(ho.get('summary', ''))
                    point = PointStruct(
                        id=hid,
                        vector={
                            "dense": vec,
                            SPARSE_VECTOR_NAME: SparseVector(
                                indices=list(sparse_vec.keys()),
                                values=list(sparse_vec.values())
                            )
                        },
                        payload=payload
                    )
                else:
                    point = PointStruct(id=hid, vector=vec, payload=payload)
                
                self.client.upsert(COLLECTION, points=[point])
        except Exception as e:
            logger.warning(f"Error creating handover: {e}")
        
        # Cleanup old turns
        await flight_recorder_service.cleanup_old_turns_async("default_user", proj)
        self.reset_session(proj)
        return "✅ Session Finalized. Flight recorder pruned."
    
    def get_handover(
        self,
        proj: str,
        format: str = "condensed",
        max_turns: int = 20
    ) -> str:
        """
        Get context for resuming a session.
        
        Formats:
        - "full": Complete turn history
        - "condensed": Summarized turns (default)
        - "minimal": Just topics + last 5 turns condensed
        """
        output: List[str] = []
        
        # Auto-detect project if not provided
        if not proj:
            most_recent_proj: Optional[str] = None
            most_recent_ts: float = 0
            if FLIGHT_DIR.exists():
                for f in FLIGHT_DIR.glob("flight_recorder_*.jsonl"):
                    try:
                        lines = f.read_text(encoding="utf-8").strip().split('\n')
                        if lines and lines[0]:
                            try:
                                last_entry = json.loads(lines[-1])
                                ts = last_entry.get('ts', 0)
                                if ts > most_recent_ts:
                                    most_recent_ts = ts
                                    p_name = f.stem.replace("flight_recorder_", "")
                                    if p_name == "global":
                                        p_name = ""
                                    most_recent_proj = p_name
                            except json.JSONDecodeError:
                                logger.warning(f"Malformed JSON in {f.name}")
                    except Exception:
                        continue
            if most_recent_proj is not None:
                proj = most_recent_proj
                output.append(f"🔄 **Auto-Detected Project:** {proj or 'Global'}")
            else:
                output.append("⚠️ No previous session found. Starting fresh.")
        
        # Read flight recorder
        turns = read_rec(proj, "default_user")
        
        if turns:
            recent_turns = turns[-max_turns:] if len(turns) > max_turns else turns
            
            if format == "full":
                context_str = format_recent_turns(recent_turns)
            elif format == "minimal":
                if len(recent_turns) <= 5:
                    context_str = self._format_turns_condensed(recent_turns)
                else:
                    last_5 = recent_turns[-5:]
                    older = recent_turns[:-5]
                    older_topics = [
                        f"• {t['turn']['user'].split('.')[0][:50]}..."
                        for t in older
                        if 'user' in t.get('turn', {})
                    ]
                    context_str = (
                        "EARLIER:\n" + "\n".join(older_topics) +
                        "\n\nRECENT:\n" + self._format_turns_condensed(last_5)
                    )
            else:  # condensed
                context_str = self._format_turns_condensed(recent_turns)
            
            output.append(f"📝 **Recent Conversation ({len(recent_turns)} turns):**\n{context_str}")
        
        # Check for stored handover
        hid = get_int_id(f"handover_{proj}", 100000000)
        try:
            res = self.client.retrieve(COLLECTION, ids=[hid])
            if res:
                ho = res[0].payload
                output.append(f"🎯 **Last Session Summary:** {ho.get('summary', 'N/A')}")
                if ho.get('last_topic'):
                    output.append(f"📌 **Last Topic:** {ho.get('last_topic')}")
                if ho.get('unresolved'):
                    output.append(f"⚠️ **Unresolved:** {ho.get('unresolved')}")
        except Exception as e:
            logger.warning(f"Error retrieving handover: {e}")
        
        return "\n\n".join(output) if output else "No context found. Fresh session."
    
    def _format_turns_condensed(self, turns: List[Dict[str, Any]]) -> str:
        """Helper for condensed formatting."""
        formatted: List[str] = []
        for t in turns:
            ts = t.get('ts', 0)
            user = t.get('turn', {}).get('user', '')
            ai = t.get('turn', {}).get('ai', '(Pending)')
            formatted.append(format_condensed_turn(user, ai, ts))
        return "\n\n".join(formatted)