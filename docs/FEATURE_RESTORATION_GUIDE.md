# Decay Memory v2.0 Feature Restoration Guide

## Purpose

This document provides a detailed guide for restoring v1.0 functionality that was lost during the Docker/cloud refactor. It maps each missing feature to its v1.0 implementation and specifies where it should go in v2.0's modular architecture.

**Source:** `decay_memory_server_v10.11.py` (95KB monolith)  
**Target:** `~/decay_memory_app/Decay_Memory_2.0/` (modular structure)

---

## Current v2.0 Module Structure

```
Decay_Memory_2.0/
├── server.py                 # FastAPI endpoints, main orchestration
├── memory_service.py         # MemorySystem class (search, add, project filter)
├── flight_recorder_service.py # Conversation logging
├── embedding_helper.py       # OpenAI embeddings
├── qdrant_client_wrapper.py  # Qdrant connection
├── soul.py                   # Emotional state machine
└── frontend/                 # React UI
```

---

## Feature Restoration Checklist

### Priority 1: Core Memory Management (CRITICAL)

These features are essential for a functional memory system.

---

#### 1.1 Memory Feedback System

**What it does:** Allows explicit user/system feedback to adjust memory importance.

**v1.0 Functions:**
```python
def boost_memory(self, memory_id: str, reason: str = "") -> str
def deprecate_memory(self, memory_id: str, reason: str = "") -> str
def correct_memory(self, memory_id: str, new_content: str) -> str
def delete_memory(self, memory_id: str, reason: str = "") -> str
def set_memory_project(self, memory_id: str, project_name: str) -> str
def set_chat_project(self, chat_id: str, project_name: str) -> str
```

**v1.0 Location:** Inside `MemorySystem` class

**Target in v2.0:** Add to `memory_service.py` → `MemorySystem` class

**Implementation Notes:**
- These modify Qdrant point payloads directly
- `boost_memory`: Multiplies `boost_factor` by 1.5 (max 5.0), records in `boost_history`
- `deprecate_memory`: Sets `deprecated=True`, records reason
- `correct_memory`: Updates `content`, regenerates embedding, records in `correction_history`
- `delete_memory`: Soft-delete by changing `type` to `archived_deleted`
- Add API endpoints in `server.py` for each

**Qdrant Payload Fields Required:**
```python
{
    "boost_factor": 1.0,        # float, max 5.0
    "boost_history": [],        # list of {reason, timestamp}
    "deprecated": False,        # bool
    "deprecation_reason": "",   # str
    "correction_history": [],   # list of {old_content, new_content, timestamp}
    "archived": False,          # bool
    "archive_reason": ""        # str
}
```

---

#### 1.2 Conflict Resolution

**What it does:** Detects when new memories contradict existing ones and resolves intelligently.

**v1.0 Functions:**
```python
def check_conflict(self, new_content: str, new_type: str, proj: str) -> Optional[Dict]
def resolve_conflict(self, old_memory: Dict, new_content: str) -> Dict
def replace_memory(self, old_id: str, new_content: str, resolution_type: str) -> str
```

**v1.0 Location:** Inside `MemorySystem` class

**Target in v2.0:** Add to `memory_service.py` → `MemorySystem` class

**Implementation Notes:**
- `check_conflict`: Searches for similar memories (similarity > 0.85), returns potential conflict
- `resolve_conflict`: Calls Gemini to analyze relationship, returns resolution type
- Resolution types: `SUPERSEDE`, `UPDATE`, `COMPLEMENT`, `UNRELATED`
- `replace_memory`: Archives old memory, stores new one with link to old

**Gemini Prompt (from v1.0):**
```python
CONFLICT_RESOLUTION_PROMPT = """
Analyze these two pieces of information about the same topic:

EXISTING: {old_content}
NEW: {new_content}

Determine the relationship:
- SUPERSEDE: New completely replaces old (old is outdated)
- UPDATE: New modifies/corrects old (merge them)
- COMPLEMENT: Both are valid (store both)
- UNRELATED: No actual conflict

Return JSON: {"resolution": "TYPE", "merged_content": "if UPDATE", "reason": "explanation"}
"""
```

**Integration Point:**
- Call `check_conflict()` at start of `add_memory()`
- If conflict found, call `resolve_conflict()` before storing

---

#### 1.3 Session Management

**What it does:** Manages conversation sessions, enables context handover between chats.

**v1.0 Functions:**
```python
def reset_session(self, proj: Optional[str], init_only: bool = False) -> None
async def finalize_async(self, summary: str, proj: str) -> str
def get_handover(self, proj: str, format: str = "condensed", max_turns: int = 20) -> Dict
def _format_turns_condensed(self, turns: List[Dict[str, Any]]) -> str
```

**v1.0 Location:** Inside `MemorySystem` class

**Target in v2.0:** Add to `memory_service.py` → `MemorySystem` class

**Implementation Notes:**
- `reset_session`: Clears session state for project, optionally initializes fresh
- `finalize_async`: 
  1. Creates summary memory (type: `context_handover`)
  2. Extracts facts via Librarian
  3. Resets session state
- `get_handover`: Returns recent flight recorder turns + last session summary
- Format options: `full`, `condensed`, `minimal`

**Session State Structure:**
```python
{
    "project_name": str,
    "turn_count": int,
    "start_time": float,
    "last_activity": float
}
```

**API Endpoints Needed:**
```python
POST /session/reset
POST /session/finalize
GET /session/handover
```

---

### Priority 2: Knowledge Graph (HIGH)

This is the biggest missing feature - enables associative recall.

---

#### 2.1 Knowledge Graph System

**What it does:** Extracts entities from memories, builds relationship graph, enables graph-based search expansion.

**v1.0 Functions:**
```python
# Standalone
def extract_knowledge_graph_data(content: str) -> Optional[Dict[str, Any]]

# KnowledgeGraphManager class
def __init__(self, memory_system: 'MemorySystem')
def invalidate_cache(self) -> None
def get_graph(self) -> nx.DiGraph
def _build_graph(self) -> nx.DiGraph
def find_connection(self, entity_a: str, entity_b: str) -> List[str]
def get_related_entities(self, entity: str, depth: int = 1) -> List[str]
def get_entity_neighborhood(self, entity: str, radius: int = 2) -> nx.DiGraph

# MemorySystem methods
def _extract_and_store_graph(self, content: str, memory_id: str, proj: str) -> None
def _expand_with_graph(self, results: List, query: str, limit: int) -> List
def find_connection(self, ...) -> Dict  # Wrapper
def get_related_entities(self, ...) -> Dict  # Wrapper
def get_entity_neighborhood(self, ...) -> Dict  # Wrapper
```

**v1.0 Location:** Standalone function + `KnowledgeGraphManager` class + `MemorySystem` methods

**Target in v2.0:** Create new `knowledge_graph_service.py`

**Implementation Notes:**

1. **Entity Extraction** uses Gemini:
```python
ENTITY_EXTRACTION_PROMPT = """
Extract entities and relationships from this text.

Text: {content}

Return JSON:
{
    "entities": ["entity1", "entity2", ...],
    "relationships": [
        {"from": "entity1", "relation": "relates_to", "to": "entity2"},
        ...
    ]
}
"""
```

2. **Graph Storage:** Entities stored as memories with `type: "knowledge_graph"`
```python
{
    "type": "knowledge_graph",
    "content": "entity_a -[relation]-> entity_b",
    "entity_from": "entity_a",
    "entity_to": "entity_b", 
    "relation": "relation_type",
    "source_memory_id": "original_memory_uuid"
}
```

3. **Graph Building:** On-demand from Qdrant, cached for 300 seconds
```python
def _build_graph(self) -> nx.DiGraph:
    # Query all knowledge_graph type memories
    # Build NetworkX DiGraph from relationships
    # Return graph
```

4. **Search Expansion:** After semantic search, expand results via graph
```python
def _expand_with_graph(self, results, query, limit):
    # Extract entities from query
    # Find related entities via graph
    # Search for memories mentioning related entities
    # Merge and dedupe results
```

**Dependencies:**
```
networkx>=3.0
```

**Integration Points:**
- Call `_extract_and_store_graph()` in `add_memory()` after storing
- Call `_expand_with_graph()` in `search()` after hybrid search
- Add API endpoints for graph queries

---

### Priority 3: Proactive Intelligence (MEDIUM)

Makes the AI smarter about when to surface memories.

---

#### 3.1 Full Proactive Retrieval

**What it does:** Intelligently detects when to surface memories without explicit request.

**v1.0 Functions:**
```python
def detect_proactive_triggers(user_message: str, ...) -> Dict[str, Any]
def generate_llm_query(user_message: str, project_name: str) -> str
def calculate_injection_score(memory: Dict, trigger_info: Dict, ...) -> float
def format_proactive_context(memories: List[Dict[str, Any]]) -> str
def proactive_retrieval(self, user_message: str, project_name: str) -> Optional[str]
def update_proactive_config(self, updates: Dict[str, Any]) -> str
```

**v1.0 Location:** Standalone functions + `MemorySystem` methods

**Target in v2.0:** Create new `proactive_service.py` OR add to `memory_service.py`

**Current v2.0 State:** Has stub `proactive_retrieval()` but missing all intelligence

**Implementation Notes:**

1. **Trigger Detection:**
```python
TRIGGER_PATTERNS = {
    "explicit": ["remember", "last time", "you mentioned", "we discussed"],
    "entity": r'\b[A-Z][a-z]+\b',  # Capitalized words
    "project": ["the project", "that issue", "the code"],
    "temporal": ["yesterday", "last week", "before"]
}
```

2. **Injection Scoring Formula:**
```python
score = (
    semantic_similarity * 0.4 +
    decay_score * 0.3 +
    type_weight * 0.2 +
    trigger_match * 0.1
)
# Inject if score > injection_threshold (default 0.15)
```

3. **LLM Query Expansion:** When user message is ambiguous, use Gemini to generate better search query

**Config Structure:**
```python
PROACTIVE_CONFIG = {
    "enabled": True,
    "trigger_threshold": 0.60,
    "max_injections": 3,
    "fast_path_threshold": 0.50,
    "injection_threshold": 0.15
}
```

---

#### 3.2 Proactive Cache

**What it does:** Prevents redundant memory retrievals within short time window.

**v1.0 Functions:**
```python
class ProactiveCache:
    def __init__(self, max_size: int, ttl: float)
    def get(self, key: str) -> Optional[Any]
    def set(self, key: str, value: Any) -> None
    def clear(self) -> None
```

**v1.0 Location:** Standalone class

**Target in v2.0:** Add to `proactive_service.py` or `memory_service.py`

**Implementation Notes:**
- TTL-based cache (default 60 seconds)
- Key = hash of (user_message, project_name)
- Prevents same query from hitting Qdrant repeatedly

---

### Priority 4: Configuration & Utilities (LOW)

Quality of life features.

---

#### 4.1 Hot-Reload Configuration

**What it does:** Allows runtime tuning without restart.

**v1.0 Functions:**
```python
def load_config() -> Optional[Dict[str, Any]]
async def load_config_async() -> Optional[Dict[str, Any]]
def get_config_value(path: str, default: Any) -> Any
```

**v1.0 Location:** Standalone functions

**Target in v2.0:** Create new `config_service.py` (separate from existing `config.py` which is env vars)

**Implementation Notes:**
- Reads `dashboard_config.json`
- Checks file timestamp on each access
- Falls back to hardcoded defaults if missing
- Config categories: decay, feedback, search, graph, flight_recorder, proactive

---

#### 4.2 Text Condensation

**What it does:** Reduces token usage for flight recorder and handover.

**v1.0 Functions:**
```python
def condense_text(text: str, aggressive: bool = False) -> str
def format_condensed_turn(user_msg: str, ai_summary: str, timestamp: float) -> str
```

**v1.0 Location:** Standalone functions

**Target in v2.0:** Add to `flight_recorder_service.py`

**Implementation Notes:**
- Removes filler words, excessive whitespace
- `aggressive=True` removes more (for older turns)
- Used in `get_handover()` with `format="condensed"`

---

#### 4.3 ID Generation

**What it does:** Deterministic ID generation for points.

**v1.0 Functions:**
```python
def get_id(seed: str, salt: str = "") -> str
def get_int_id(seed: str, offset: int) -> int
```

**v1.0 Location:** Standalone functions

**Target in v2.0:** Add to `memory_service.py` or new `utils.py`

**Implementation Notes:**
- Uses MD5 hash for deterministic IDs
- `get_id`: Returns UUID string
- `get_int_id`: Returns int for Qdrant point IDs

---

#### 4.4 Point Creation Helpers

**What it does:** Creates properly structured Qdrant points.

**v1.0 Functions:**
```python
def create_point(content, mtype, proj, ...) -> PointStruct
def create_empty_point(point_id: int, payload: Dict) -> PointStruct
```

**v1.0 Location:** Standalone functions

**Target in v2.0:** Add to `memory_service.py`

**Implementation Notes:**
- `create_point`: Full point with vectors, payload, timestamp
- `create_empty_point`: For graph entries (no vector needed)
- Ensures consistent payload structure

---

#### 4.5 Project List Cache

**What it does:** Caches list of projects to avoid repeated Qdrant queries.

**v1.0 Functions:**
```python
class ProjectListCache:
    def __init__(self, ttl: float)
    def get(self) -> Optional[List[str]]
    def set(self, projects: List[str]) -> None
    def invalidate(self) -> None
```

**v1.0 Location:** Standalone class

**Target in v2.0:** Add to `memory_service.py`

**Implementation Notes:**
- TTL-based (default from config)
- Invalidated when new project is created
- Called by `list_projects()`

---

## Restoration Order (Recommended)

```
Week 1-2: Priority 1 (Core Memory Management)
├── 1.1 Memory Feedback System
├── 1.2 Conflict Resolution  
└── 1.3 Session Management

Week 3-4: Priority 2 (Knowledge Graph)
└── 2.1 Full Knowledge Graph System

Week 5: Priority 3 (Proactive Intelligence)
├── 3.1 Full Proactive Retrieval
└── 3.2 Proactive Cache

Week 6: Priority 4 (Configuration & Utilities)
├── 4.1 Hot-Reload Configuration
├── 4.2 Text Condensation
├── 4.3 ID Generation
├── 4.4 Point Creation Helpers
└── 4.5 Project List Cache
```

---

## API Endpoints to Add

After restoration, these endpoints should exist:

```python
# Memory Feedback
POST /memory/{id}/boost
POST /memory/{id}/deprecate
POST /memory/{id}/correct
DELETE /memory/{id}
PUT /memory/{id}/project

# Session Management
POST /session/reset
POST /session/finalize
GET /session/handover

# Knowledge Graph
GET /graph/connection?a={entity}&b={entity}
GET /graph/related?entity={entity}
GET /graph/neighborhood?entity={entity}

# Configuration
GET /config
PUT /config
```

---

## Testing Checklist

After each feature restoration:

- [ ] Unit tests pass
- [ ] Integration with existing v2.0 code works
- [ ] Multi-user support maintained (user_id parameter)
- [ ] API endpoint responds correctly
- [ ] Docker container builds
- [ ] Frontend can access (if applicable)

---

## Files to Reference

When porting, these are the key sections of `decay_memory_server_v10.11.py`:

| Feature | Approximate Lines |
|---------|-------------------|
| Configuration | 1-200 |
| Flight Recorder | 200-400 |
| Embeddings | 400-500 |
| Sparse Vectors | 500-600 |
| Knowledge Graph | 600-900 |
| Proactive Retrieval | 900-1200 |
| MemorySystem class | 1200-2500 |
| MCP Tools | 2500-end |

---

*Document created: December 5, 2025*
*Purpose: Guide for Cursor/Opus to restore v1.0 functionality to v2.0*
