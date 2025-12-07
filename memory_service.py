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
import networkx as nx

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

# Knowledge Graph Constants (from v1.0)
GRAPH_CACHE_SECONDS = 300  # Rebuild graph every 5 minutes
GRAPH_AUTO_INVALIDATE = True  # Auto-invalidate on memory changes
MAX_SCROLL_LIMIT = 10000  # Max memories to scan for graph building

# Proactive Retrieval Constants (from v1.0)
EXPLICIT_MEMORY_KEYWORDS = [
    "remember", "recall", "last time", "yesterday", "before", 
    "earlier", "we discussed", "you said", "you mentioned"
]
PROJECT_CONTINUITY_KEYWORDS = [
    "the project", "that issue", "this feature", "it", "that",
    "the system", "the code", "the script", "the implementation"
]
GREETING_KEYWORDS = [
    "good morning", "good afternoon", "good evening",
    "hi", "hello", "hey there"
]
QUESTION_STARTERS = [
    "what", "when", "where", "how", "why", "who", "which", "can", "should", "is", "are"
]
MAX_ENTITIES_TO_SEARCH = 3
MAX_MEMORY_CONTENT_PREVIEW = 200

# Memory half-lives by type (seconds)
HALF_LIVES = {
    'personal': 2592000,      # 30 days
    'preference': 1296000,    # 15 days  
    'goal': 604800,           # 7 days
    'project': 432000,        # 5 days
    'topic': 259200,          # 3 days
    'dialog': 86400,          # 1 day
    'context_handover': 259200  # 3 days
}

# Type weights for injection scoring
TYPE_WEIGHTS = {
    'project': 1.5, 'goal': 1.3, 'personal': 1.2, 'topic': 1.0,
    'preference': 0.9, 'dialog': 0.8, 'context_handover': 1.4
}

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


# =============================================================================
# KNOWLEDGE GRAPH EXTRACTION (Restored from v1.0)
# =============================================================================

def extract_knowledge_graph_data(content: str) -> Optional[Dict[str, Any]]:
    """Extract entities and relationships using Gemini."""
    import os
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
    except ImportError:
        return None
    
    prompt = (
        f"Extract entities and relationships from this text. "
        f"Return JSON: {{'entities': [{{'name':..., 'type':...}}], "
        f"'relationships': [{{'from':..., 'to':..., 'type':...}}]}}\n\n"
        f"Text: {content}"
    )
    try:
        res = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config={"response_mime_type": "application/json"}
        ).generate_content(prompt)
        data = json.loads(res.text)
        return data if data.get('entities') or data.get('relationships') else None
    except Exception as e:
        logger.warning(f"Graph extraction failed: {e}")
        return None


# =============================================================================
# KNOWLEDGE GRAPH BUILDER (Restored from v1.0)
# =============================================================================

class KnowledgeGraphBuilder:
    """Builds and queries the knowledge graph from stored memories."""
    
    def __init__(self, client):
        self.client = client
        self.graph: Optional[nx.DiGraph] = None
        self.last_build: float = 0
    
    def invalidate_cache(self) -> None:
        """Force graph rebuild on next access."""
        self.last_build = 0
    
    def get_graph(self) -> nx.DiGraph:
        """Get or rebuild the knowledge graph."""
        if not self.graph or (time.time() - self.last_build) > GRAPH_CACHE_SECONDS:
            self.graph = self._build_graph()
            self.last_build = time.time()
        return self.graph
    
    def _build_graph(self) -> nx.DiGraph:
        """Build graph from knowledge_graph type memories."""
        G = nx.DiGraph()
        try:
            filter_kg = Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="knowledge_graph"))
            ])
            mems, _ = self.client.scroll(
                COLLECTION,
                scroll_filter=filter_kg,
                limit=MAX_SCROLL_LIMIT,
                with_payload=True
            )
            for m in mems:
                try:
                    d = json.loads(m.payload.get('content', '{}'))
                except json.JSONDecodeError:
                    continue
                    
                age = time.time() - m.payload.get('timestamp', 0)
                # Half-life of 180 days for graph nodes
                score = max(0.5**(age/15552000), 0.3)
                
                for e in d.get('entities', []):
                    if not e.get('name'):
                        continue
                    if not G.has_node(e['name']):
                        G.add_node(e['name'], type=e.get('type'), decay_score=score)
                        
                for r in d.get('relationships', []):
                    if r.get('from') and r.get('to'):
                        G.add_edge(r['from'], r['to'], type=r.get('type'), decay_score=score)
                        
            logger.info(f"Knowledge graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Error building graph: {e}")
        return G
    
    def find_connection(
        self,
        entity_a: str,
        entity_b: str,
        max_hops: int = 3
    ) -> Dict[str, Any]:
        """Find shortest path between two entities."""
        G = self.get_graph()
        matches_a = [n for n in G.nodes() if entity_a.lower() in n.lower()]
        matches_b = [n for n in G.nodes() if entity_b.lower() in n.lower()]
        
        if not matches_a:
            return {"error": f"Entity '{entity_a}' not found"}
        if not matches_b:
            return {"error": f"Entity '{entity_b}' not found"}
        
        best_path: Optional[Dict[str, Any]] = None
        best_strength: float = 0
        
        for ma in matches_a:
            for mb in matches_b:
                try:
                    path = nx.shortest_path(G, ma, mb)
                    if len(path) - 1 > max_hops:
                        continue
                    
                    node_scores = [G.nodes[n].get('decay_score', 1.0) for n in path]
                    edge_scores = [
                        G[path[i]][path[i+1]].get('decay_score', 1.0) 
                        for i in range(len(path)-1)
                    ]
                    path_strength = min(node_scores + edge_scores)
                    
                    if path_strength > best_strength:
                        best_strength = path_strength
                        best_path = {
                            "path": path,
                            "relationships": [
                                {
                                    "from": path[i],
                                    "to": path[i+1],
                                    "type": G[path[i]][path[i+1]].get('type', 'unknown')
                                }
                                for i in range(len(path)-1)
                            ],
                            "strength": round(path_strength, 3),
                            "hops": len(path) - 1
                        }
                except Exception:
                    continue
        
        if not best_path:
            return {"error": f"No path found between '{entity_a}' and '{entity_b}'"}
        return best_path
    
    def get_related_entities(
        self,
        entity: str,
        relationship_type: Optional[str] = None,
        depth: int = 1
    ) -> Dict[str, Any]:
        """Get entities related to the given entity."""
        G = self.get_graph()
        matches = [n for n in G.nodes() if entity.lower() in n.lower()]
        if not matches:
            return {"error": f"Entity '{entity}' not found"}
        
        results: List[Dict[str, Any]] = []
        for entity_name in matches:
            if depth == 1:
                neighbors = list(G.neighbors(entity_name))
            else:
                neighbors = list(
                    nx.single_source_shortest_path_length(G, entity_name, cutoff=depth).keys()
                )
                if entity_name in neighbors:
                    neighbors.remove(entity_name)
            
            related: List[Dict[str, Any]] = []
            for neighbor in neighbors:
                try:
                    edge_data = G[entity_name].get(neighbor, {})
                    if relationship_type and edge_data.get('type') != relationship_type:
                        continue
                    related.append({
                        "entity": neighbor,
                        "entity_type": G.nodes[neighbor].get('type'),
                        "relationship": edge_data.get('type', 'unknown'),
                        "strength": round(edge_data.get('decay_score', 1.0), 3)
                    })
                except Exception:
                    continue
            
            results.append({
                "entity": entity_name,
                "entity_type": G.nodes[entity_name].get('type'),
                "related_count": len(related),
                "related_entities": sorted(related, key=lambda x: x['strength'], reverse=True)
            })
        
        return {"results": results}
    
    def get_entity_neighborhood(
        self,
        entity: str,
        radius: int = 2
    ) -> Dict[str, Any]:
        """Get the subgraph around an entity."""
        G = self.get_graph()
        matches = [n for n in G.nodes() if entity.lower() in n.lower()]
        if not matches:
            return {"error": f"Entity '{entity}' not found"}
        
        try:
            subgraph = nx.ego_graph(G, matches[0], radius=radius)
        except Exception as e:
            return {"error": f"Failed to build neighborhood: {e}"}
        
        return {
            "center_entity": matches[0],
            "radius": radius,
            "node_count": subgraph.number_of_nodes(),
            "edge_count": subgraph.number_of_edges(),
            "nodes": [
                {
                    "name": n,
                    "type": subgraph.nodes[n].get('type'),
                    "decay_score": round(subgraph.nodes[n].get('decay_score', 1.0), 3)
                }
                for n in subgraph.nodes()
            ],
            "edges": [
                {
                    "from": u,
                    "to": v,
                    "type": subgraph[u][v].get('type'),
                    "decay_score": round(subgraph[u][v].get('decay_score', 1.0), 3)
                }
                for u, v in subgraph.edges()
            ]
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        G = self.get_graph()
        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "is_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
            "density": round(nx.density(G), 4) if G.number_of_nodes() > 0 else 0,
            "node_types": dict(Counter(
                G.nodes[n].get('type', 'unknown') for n in G.nodes()
            )),
            "relationship_types": dict(Counter(
                G[u][v].get('type', 'unknown') for u, v in G.edges()
            ))
        }


# =============================================================================
# PROACTIVE RETRIEVAL FUNCTIONS (Restored from v1.0)
# =============================================================================

def detect_proactive_triggers(
    user_message: str,
    project_name: str,
    time_since_last: float
) -> List[Dict[str, Any]]:
    """Detect if proactive retrieval should fire based on message patterns."""
    triggers: List[Dict[str, Any]] = []
    message_lower = user_message.lower()
    
    # Check for explicit memory references
    if any(kw in message_lower for kw in EXPLICIT_MEMORY_KEYWORDS):
        triggers.append({
            "type": "explicit_memory",
            "confidence": 0.90,
            "query": user_message,
            "reason": "User explicitly referenced past memory"
        })
    
    # Entity detection (capitalized words)
    words = user_message.split()
    entities = [w for w in words if len(w) > 2 and w[0].isupper() and w not in ["I", "The", "A", "An"]]
    
    if entities:
        for entity in entities[:MAX_ENTITIES_TO_SEARCH]:
            triggers.append({
                "type": "entity_mention",
                "confidence": 0.75,
                "query": entity,
                "graph_expand": True,
                "reason": f"Entity mentioned: {entity}"
            })
    
    # Project continuity detection
    has_project_ref = any(ref in message_lower for ref in PROJECT_CONTINUITY_KEYWORDS)
    if has_project_ref and project_name:
        triggers.append({
            "type": "project_continuity",
            "confidence": 0.70,
            "query": f"recent work {project_name}",
            "project_name": project_name,
            "reason": "Reference to ongoing project work"
        })
    
    # Greeting after time gap
    min_time_gap_hours = PROACTIVE_CONFIG.get("min_time_gap_hours", 4)
    is_greeting = any(g in message_lower for g in GREETING_KEYWORDS)
    long_gap = time_since_last > (min_time_gap_hours * 3600)
    
    if is_greeting and long_gap:
        triggers.append({
            "type": "time_based_greeting",
            "confidence": 0.65,
            "query": f"last session {project_name}" if project_name else "last session",
            "project_name": project_name,
            "reason": f"Greeting after {time_since_last/3600:.1f}h gap"
        })
    
    # Question pattern
    is_question = any(message_lower.startswith(q) for q in QUESTION_STARTERS) or "?" in user_message
    
    if is_question and project_name:
        triggers.append({
            "type": "question_pattern",
            "confidence": 0.65,
            "query": user_message,
            "project_name": project_name,
            "needs_llm": True,
            "reason": "Question that might reference context"
        })
    
    if triggers:
        logger.debug(f"Proactive triggers detected: {[t['type'] for t in triggers]}")
    
    return triggers


def generate_llm_query(user_message: str, project_name: str) -> str:
    """Use Gemini to generate a better search query."""
    import os
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return user_message
        genai.configure(api_key=api_key)
    except ImportError:
        return user_message
    
    prompt = f"""Given this user message: "{user_message}"
Current project: {project_name or "None"}

Generate a concise search query (2-6 words) to find the most relevant memories.
Focus on what the user is actually asking about or referencing.

Return ONLY the search query, nothing else."""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        result = model.generate_content(prompt)
        query = result.text.strip().strip('"').strip("'")
        return query
    except Exception as e:
        logger.warning(f"LLM Query Gen failed: {e}")
        return user_message


def calculate_injection_score(
    memory: Dict[str, Any],
    user_message: str,
    trigger_confidence: float
) -> float:
    """Calculate whether a retrieved memory should be injected into context."""
    semantic_score = memory.get('search_score', memory.get('score', 0.5))
    
    # Calculate decay score
    age_seconds = time.time() - memory.get('timestamp', 0)
    half_life = HALF_LIVES.get(memory.get('type'), 604800)
    decay_score = max(0.5 ** (age_seconds / half_life), DECAY_FLOOR)
    
    # Recency boost (recent memories get a bump)
    age_hours = age_seconds / 3600
    recency_boost = 1.5 if age_hours < 24 else (1.2 if age_hours < 72 else 1.0)
    
    # Type-specific weight
    type_weight = TYPE_WEIGHTS.get(memory.get('type'), 1.0)
    
    # User-applied boost factor
    boost = memory.get('boost_factor', 1.0)
    
    # Final injection score
    injection_score = (
        semantic_score * decay_score * recency_boost * 
        type_weight * boost * trigger_confidence
    )
    
    return injection_score


def format_proactive_context(memories: List[Dict[str, Any]]) -> str:
    """Format retrieved memories for injection into AI context."""
    if not memories:
        return ""
    
    context = "## 🧠 PROACTIVE CONTEXT\n\n"
    context += "*(Retrieved automatically based on conversation analysis)*\n\n"
    
    max_injections = PROACTIVE_CONFIG.get("max_injections", 3)
    for i, mem in enumerate(memories[:max_injections], 1):
        age_hours = (time.time() - mem.get('timestamp', 0)) / 3600
        if age_hours < 1:
            age = "just now"
        elif age_hours < 24:
            age = f"{age_hours:.0f}h ago"
        elif age_hours < 168:
            age = f"{age_hours/24:.0f}d ago"
        else:
            age = f"{age_hours/168:.0f}w ago"
        
        content = mem.get('content', '')[:MAX_MEMORY_CONTENT_PREVIEW]
        if len(mem.get('content', '')) > MAX_MEMORY_CONTENT_PREVIEW:
            content += "..."
        
        mem_type = mem.get('type', 'unknown')
        score = mem.get('injection_score', 0.0)
        
        context += f"{i}. **[{age}]** {content}\n"
        context += f"   *Type: {mem_type} | Relevance: {score:.2f}*\n\n"
    
    context += "---\n"
    context += "*Use this context naturally if relevant. Don't mention retrieval unless asked.*\n\n"
    
    return context

class MemorySystem:
    def __init__(self):
        # 1. Initialize DB via Wrapper
        self.client = qdrant_client_wrapper.get_qdrant_client()
        qdrant_client_wrapper.init_db(self.client)
        
        # 2. Initialize Knowledge Graph Builder
        self.graph_builder = KnowledgeGraphBuilder(self.client)
        
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
        
        # Extract and store knowledge graph data
        self._extract_and_store_graph(content, proj, pid)
        
        return "Memory saved."
    
    def _extract_and_store_graph(self, content: str, proj: str, source_memory_id: str) -> None:
        """Extract entities/relationships and store as knowledge graph memory."""
        graph_data = extract_knowledge_graph_data(content)
        if not graph_data:
            return
        
        import uuid
        dense_vec = embedding_helper.embed(content)
        if not dense_vec:
            return
        
        pid = str(uuid.uuid4())
        payload = {
            "content": json.dumps(graph_data),
            "type": "knowledge_graph",
            "project_name": proj,
            "source_memory_id": source_memory_id,
            "timestamp": time.time(),
            "entity_count": len(graph_data.get('entities', [])),
            "relationship_count": len(graph_data.get('relationships', []))
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
        
        if GRAPH_AUTO_INVALIDATE:
            self.graph_builder.invalidate_cache()
        
        logger.info(f"Stored graph: {len(graph_data.get('entities', []))} entities, {len(graph_data.get('relationships', []))} relationships")

    # --- API Wrapper ---
    def proactive_retrieval(
        self, 
        user_id: str, 
        user_message: str, 
        project_name: str,
        time_since_last: float = 0
    ) -> Optional[str]:
        """
        Full proactive retrieval with trigger detection and injection scoring.
        
        1. Detect triggers (explicit memory refs, entities, project refs, etc.)
        2. Execute searches for each trigger
        3. Score results for injection worthiness
        4. Format and return context for AI
        """
        if not PROACTIVE_CONFIG.get("enabled", True):
            return None
        
        # Step 1: Detect triggers
        triggers = detect_proactive_triggers(user_message, project_name, time_since_last)
        
        if not triggers:
            # Fast path: just do semantic search if no triggers
            results = self.hybrid_search(user_message, proj=project_name)
            valid = [m for m in results if m['search_score'] >= PROACTIVE_CONFIG['fast_path_threshold']]
            if valid:
                for m in valid:
                    m['injection_score'] = calculate_injection_score(m, user_message, 0.5)
                valid = [m for m in valid if m['injection_score'] >= PROACTIVE_CONFIG['injection_threshold']]
                if valid:
                    return format_proactive_context(valid)
            return None
        
        # Step 2: Execute searches for each trigger
        all_memories: Dict[str, Dict[str, Any]] = {}  # Dedupe by content
        
        for trigger in triggers:
            query = trigger.get('query', user_message)
            
            # Optional LLM query refinement
            if trigger.get('needs_llm'):
                query = generate_llm_query(query, project_name)
            
            # Execute search
            results = self.hybrid_search(query, proj=project_name)
            
            # Expand with graph if requested
            if trigger.get('graph_expand') and results:
                results = self._expand_with_graph(results, project_name)
            
            # Step 3: Score each result
            for mem in results:
                content = mem.get('content', '')
                if content not in all_memories:
                    injection_score = calculate_injection_score(
                        mem, user_message, trigger['confidence']
                    )
                    mem['injection_score'] = injection_score
                    mem['trigger_type'] = trigger['type']
                    all_memories[content] = mem
                else:
                    # Update score if higher
                    new_score = calculate_injection_score(
                        mem, user_message, trigger['confidence']
                    )
                    if new_score > all_memories[content].get('injection_score', 0):
                        mem['injection_score'] = new_score
                        mem['trigger_type'] = trigger['type']
                        all_memories[content] = mem
        
        # Step 4: Filter by threshold and sort
        threshold = PROACTIVE_CONFIG.get('injection_threshold', 0.15)
        valid_memories = [
            m for m in all_memories.values()
            if m.get('injection_score', 0) >= threshold
        ]
        
        if not valid_memories:
            return None
        
        # Sort by injection score
        valid_memories.sort(key=lambda x: x.get('injection_score', 0), reverse=True)
        
        # Step 5: Format
        return format_proactive_context(valid_memories)
    
    def _expand_with_graph(
        self,
        results: List[Dict[str, Any]],
        proj: str
    ) -> List[Dict[str, Any]]:
        """Expand search results using knowledge graph connections."""
        expanded = list(results)
        
        # Extract entities from results
        for mem in results[:3]:  # Check top 3
            content = mem.get('content', '')
            words = content.split()
            entities = [w for w in words if len(w) > 2 and w[0].isupper() and w not in ["I", "The", "A", "An"]]
            
            for entity in entities[:2]:
                related = self.graph_builder.get_related_entities(entity, depth=1)
                if 'results' in related:
                    for r in related['results']:
                        for rel_entity in r.get('related_entities', [])[:2]:
                            # Search for memories about related entities
                            entity_results = self.hybrid_search(rel_entity['entity'], proj=proj)
                            for er in entity_results[:1]:
                                if er.get('content') not in [e.get('content') for e in expanded]:
                                    er['graph_expanded'] = True
                                    expanded.append(er)
        
        return expanded

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

    # =========================================================================
    # KNOWLEDGE GRAPH (Restored from v1.0)
    # =========================================================================
    
    def graph_find_connection(self, entity_a: str, entity_b: str, max_hops: int = 3) -> Dict[str, Any]:
        """Find shortest path between two entities in the knowledge graph."""
        return self.graph_builder.find_connection(entity_a, entity_b, max_hops)
    
    def graph_get_related(self, entity: str, relationship_type: Optional[str] = None, depth: int = 1) -> Dict[str, Any]:
        """Get entities related to a given entity."""
        return self.graph_builder.get_related_entities(entity, relationship_type, depth)
    
    def graph_get_neighborhood(self, entity: str, radius: int = 2) -> Dict[str, Any]:
        """Get the subgraph around an entity."""
        return self.graph_builder.get_entity_neighborhood(entity, radius)
    
    def graph_get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return self.graph_builder.get_graph_stats()
    
    def graph_invalidate(self) -> str:
        """Force rebuild of the knowledge graph on next access."""
        self.graph_builder.invalidate_cache()
        return "✅ Knowledge graph cache invalidated"