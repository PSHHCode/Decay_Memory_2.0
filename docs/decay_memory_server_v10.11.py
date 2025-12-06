"""
Decay Memory MCP Server - V10.11 (Concurrency Fixes)
STATUS: PRODUCTION READY
FEATURES:
- All features from V10.10 (PRESERVED)
- FIX: Zombie lock eviction bug - reverted to simple defaultdict
- FIX: Thread-bypass race condition in finalize - now async
- FIX: Redundant config locking - single threading.Lock
- FIX: Fallthrough search - single query with combined filter

CHANGELOG V10.11:
- FIX: Removed LRULockDict - asyncio.Lock eviction while held caused data corruption
- FIX: finalize() -> finalize_async() - now respects asyncio file locks
- FIX: Removed _config_async_lock - single threading.Lock wrapped in run_in_executor
- PERF: hybrid_search now uses single query with Filter(should=[project, global])
       instead of sequential fallthrough (halves latency)
- PRESERVED: All V10.10 embedding cache, project cache, type hints, constants

CHANGELOG V10.10 (Previous):
- FIX: Race condition - sync load_config now uses threading lock
- FIX: check_conflict() filter construction
- PERF: embed() LRU cache, list_projects() caching, timestamp/chat_id indexes
- REFACTOR: create_point() helper, comprehensive type hints, named constants
"""
import time
import json
import asyncio
import os
import hashlib
import uuid
import sys
import re
import logging
import threading
import functools
from typing import Optional, List, Dict, Any, Tuple, Set
from pathlib import Path
from collections import Counter, defaultdict

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("DecayMemory")

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).parent.absolute()
ENV_PATH = SCRIPT_DIR / ".env"
CONFIG_FILE = SCRIPT_DIR / "dashboard_config.json"
FLIGHT_DIR = SCRIPT_DIR / "flight_recorders"

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
        logger.info(f"Loaded environment variables from {ENV_PATH}")
    else:
        logger.warning(f".env file not found at {ENV_PATH}")
        logger.info("Create a .env file with your API keys. See .env.example for template.")
except ImportError:
    logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")
    logger.info("Falling back to system environment variables only.")

import google.generativeai as genai
from openai import OpenAI
from mcp.server import Server
from mcp.types import Tool, TextContent
from qdrant_client import QdrantClient as OfficialQdrantClient
from qdrant_client.http.models import (
    PointStruct, Distance, VectorParams, ScalarQuantization, ScalarQuantizationConfig,
    Filter, FieldCondition, MatchValue, IsNullCondition, PayloadField,
    SparseVector, SparseVectorParams, SparseIndexParams,
    Prefetch, Fusion, FusionQuery, NamedVector
)
import networkx as nx


# =============================================================================
# CONSTANTS - Named values replacing magic numbers
# =============================================================================

# Cache size limits
MAX_PROACTIVE_CACHE_SIZE = 500
MAX_EMBEDDING_CACHE_SIZE = 1000
MAX_PROJECT_CACHE_AGE_SECONDS = 60

# Search/retrieval limits
MAX_ENTITIES_TO_SEARCH = 2
MAX_RELATED_ENTITIES_PER_RESULT = 3
MAX_GRAPH_ENTITIES_TO_EXPAND = 1
PREFETCH_MULTIPLIER = 2
MAX_SCROLL_LIMIT = 10000

# Conflict detection
CONFLICT_SIMILARITY_THRESHOLD = 0.85
CONFLICT_CONFIDENCE_THRESHOLD = 0.8

# Content truncation
MAX_MEMORY_CONTENT_PREVIEW = 150
MAX_TURN_CONTENT_LENGTH = 500
MAX_RAW_SUMMARY_LENGTH = 500

# Proactive retrieval
PROACTIVE_CACHE_TTL = 60  # seconds


# =============================================================================
# CONCURRENCY LOCKS (V10.11: Simplified - no LRU eviction)
# =============================================================================

# V10.11: Simple defaultdict - LRU eviction of held locks caused corruption
# Memory footprint is negligible (~100 bytes per lock)
_file_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def get_file_lock(filename: str) -> asyncio.Lock:
    """Get or create a lock for a specific file."""
    return _file_locks[filename]


# =============================================================================
# PROACTIVE RETRIEVAL CACHE (Thread-safe LRU with TTL)
# =============================================================================

class LRUCache:
    """Simple LRU cache with TTL support for proactive retrieval results."""
    
    def __init__(self, max_size: int = MAX_PROACTIVE_CACHE_SIZE, ttl: float = PROACTIVE_CACHE_TTL):
        from collections import OrderedDict
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            
            timestamp, value = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]
            
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[key] = (time.time(), value)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_proactive_cache = LRUCache(MAX_PROACTIVE_CACHE_SIZE, PROACTIVE_CACHE_TTL)


# =============================================================================
# CONFIG LOADER (V10.11: Single threading.Lock for both sync and async)
# =============================================================================

_config_lock = threading.Lock()
CONFIG_CACHE: Dict[str, Any] = {"data": None, "mtime": 0.0}


def load_config() -> Optional[Dict[str, Any]]:
    """Load configuration from JSON file with hot-reload support (thread-safe)."""
    global CONFIG_CACHE
    
    with _config_lock:
        if not CONFIG_FILE.exists():
            return None
        
        try:
            current_mtime = CONFIG_FILE.stat().st_mtime
            if CONFIG_CACHE["data"] and CONFIG_CACHE["mtime"] == current_mtime:
                return CONFIG_CACHE["data"]
            
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            CONFIG_CACHE["data"] = config
            CONFIG_CACHE["mtime"] = current_mtime
            logger.info(f"Loaded dashboard config (modified: {current_mtime})")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return CONFIG_CACHE.get("data")


async def load_config_async() -> Optional[Dict[str, Any]]:
    """Async wrapper - delegates to sync version via executor to respect threading.Lock."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, load_config)


def get_config_value(path: str, default: Any) -> Any:
    """Get a config value by path (e.g., 'decay_system.decay_floor')."""
    config = load_config()
    if not config:
        return default
    
    keys = path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# =============================================================================
# CONFIGURATION
# =============================================================================

COLLECTION = "decay_memory_mcp"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Phase 4A Hybrid Search Config
SPARSE_VECTOR_NAME = "keywords"
ENABLE_HYBRID_SEARCH = True

# V10.11: Removed FALLTHROUGH_THRESHOLD - now using combined filter approach

# Stop words to prevent noise in keyword search
STOP_WORDS: Set[str] = {
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'it', 'that', 'this',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
    'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'now'
}

# Sliding Window Config
FLIGHT_RECORDER_RETENTION_HOURS = get_config_value('flight_recorder.retention_hours', 48)
FLIGHT_RECORDER_MAX_TURNS = get_config_value('flight_recorder.max_turns', 100)

# Memory type classifications
GLOBAL_TYPES: Set[str] = {'personal', 'preference', 'goal'}
CONFLICT_TYPES: Set[str] = {'personal', 'preference', 'goal', 'project'}
GRAPH_EXTRACTION_TYPES: Set[str] = {'personal', 'project', 'topic'}

# Load HALF_LIVES from config with defaults
config_half_lives = get_config_value('decay_system.half_lives', {})
HALF_LIVES: Dict[str, int] = {
    'personal': config_half_lives.get('personal', 31536000),
    'preference': config_half_lives.get('preference', 15552000),
    'goal': config_half_lives.get('goal', 7776000),
    'project': config_half_lives.get('project', 2592000),
    'topic': config_half_lives.get('topic', 31536000),
    'context_handover': config_half_lives.get('context_handover', 2592000),
    'dialog': config_half_lives.get('dialog', 604800),
    'knowledge_graph': config_half_lives.get('knowledge_graph', 15552000)
}

# Configurable constants
DECAY_FLOOR = get_config_value('decay_system.decay_floor', 0.3)

# Feedback Config
BOOST_INCREMENT = get_config_value('feedback_system.boost_increment', 1.5)
DEPRECATION_MULTIPLIER = get_config_value('feedback_system.deprecation_multiplier', 0.1)
MAX_BOOST = get_config_value('feedback_system.max_boost', 5.0)

# Graph Config
GRAPH_CACHE_SECONDS = get_config_value('knowledge_graph.cache_seconds', 300)
GRAPH_AUTO_INVALIDATE = get_config_value('knowledge_graph.auto_invalidate', True)

# --- API KEY LOADING ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients (will be None if keys missing, checked in main())
openai_client: Optional[OpenAI] = None
gemini_configured: bool = False

# --- PROACTIVE RETRIEVAL CONFIG ---
config_proactive = get_config_value('proactive_retrieval', {})
PROACTIVE_CONFIG: Dict[str, Any] = {
    "enabled": config_proactive.get('enabled', True),
    "trigger_threshold": config_proactive.get('trigger_threshold', 0.50),
    "max_injections": config_proactive.get('max_injections', 3),
    "include_graph": config_proactive.get('include_graph', True),
    "min_time_gap_hours": config_proactive.get('min_time_gap_hours', 12),
    "use_llm_expansion": config_proactive.get('use_llm_expansion', True),
    "fast_path_threshold": config_proactive.get('fast_path_threshold', 0.60),
    "injection_threshold": config_proactive.get('injection_threshold', 0.30)
}

# Trigger keywords
EXPLICIT_MEMORY_KEYWORDS: List[str] = [
    "remember", "recall", "last time", "yesterday", "before", 
    "earlier", "we discussed", "you said", "you mentioned"
]

PROJECT_CONTINUITY_KEYWORDS: List[str] = [
    "the project", "that issue", "this feature", "it", "that",
    "the system", "the code", "the script", "the implementation"
]

GREETING_KEYWORDS: List[str] = [
    "good morning", "good afternoon", "good evening",
    "hi", "hello", "hey there"
]

QUESTION_STARTERS: List[str] = [
    "what", "when", "where", "how", "why", "who", "which", "can", "should", "is", "are"
]


# =============================================================================
# TOKEN REDUCTION HELPERS
# =============================================================================

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
    
    condensed = re.sub(r'\s+', ' ', condensed).strip()
    
    if aggressive:
        condensed = condensed.replace('for example', 'e.g.').replace('that is', 'i.e.')
        condensed = condensed.replace('and so on', 'etc.')
    
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


# =============================================================================
# FILENAME SANITIZATION (Security)
# =============================================================================

def get_safe_filename(proj: Optional[str]) -> str:
    """Sanitize project name to prevent path traversal attacks."""
    name = proj or "global"
    # Remove any character that isn't alphanumeric, space, hyphen, or underscore
    name = re.sub(r'[^\w\s-]', '_', name)
    name = name.strip().replace(' ', '_')
    # Extra safety: reject anything that looks like path traversal
    if '..' in name or name.startswith('/') or name.startswith('\\'):
        logger.warning(f"Detected potentially malicious project name: {proj}")
        return "flight_recorder_sanitized_input.jsonl"
    return f"flight_recorder_{name}.jsonl"


def get_rec_path(proj: Optional[str]) -> Path:
    """Get flight recorder path with sanitized filename."""
    return FLIGHT_DIR / get_safe_filename(proj)


# =============================================================================
# FILE I/O OPERATIONS (Async-first with proper locking)
# =============================================================================

def _append_to_file(path: Path, content: str) -> None:
    """Helper to append content to file (sync)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)


def _read_file_lines(path: Path) -> List[str]:
    """Helper to read file lines (sync)."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip().split('\n')


def _write_file(path: Path, content: str) -> None:
    """Helper to write file content (sync)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


async def log_turn_async(proj: str, user: str, ai: str) -> None:
    """Log a conversation turn with file locking."""
    path = get_rec_path(proj)
    filename = path.name
    async with get_file_lock(filename):
        try:
            loop = asyncio.get_running_loop()
            line = json.dumps({"ts": time.time(), "turn": {"user": user, "ai": ai}}) + "\n"
            await loop.run_in_executor(None, lambda: _append_to_file(path, line))
        except Exception as e:
            logger.error(f"Error logging turn: {e}")


async def update_last_turn_response_async(proj: str, ai_response: str) -> bool:
    """Update the most recent flight recorder entry with file locking."""
    path = get_rec_path(proj)
    filename = path.name
    async with get_file_lock(filename):
        try:
            if not path.exists():
                return False
            
            loop = asyncio.get_running_loop()
            
            def _process_file() -> bool:
                lines = _read_file_lines(path)
                if not lines or lines == ['']:
                    return False
                
                try:
                    last_entry = json.loads(lines[-1])
                except json.JSONDecodeError:
                    logger.warning(f"Malformed last line in {filename}, skipping update")
                    return False
                
                if last_entry.get('turn', {}).get('ai') != "(AI Response Pending)":
                    return False
                
                last_entry['turn']['ai'] = ai_response
                lines[-1] = json.dumps(last_entry)
                
                content = '\n'.join(lines) + '\n'
                _write_file(path, content)
                return True
            
            return await loop.run_in_executor(None, _process_file)
            
        except Exception as e:
            logger.error(f"Error updating last turn: {e}")
            return False


def read_rec(proj: str) -> List[Dict[str, Any]]:
    """Read flight recorder with malformed line handling."""
    p = get_rec_path(proj)
    if not p.exists():
        return []
    turns: List[Dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line {line_num} in {p.name}")
    except Exception as e:
        logger.error(f"Error reading flight recorder: {e}")
    return turns


async def cleanup_old_turns_async(proj: str) -> None:
    """Prune old logs with file locking."""
    path = get_rec_path(proj)
    filename = path.name
    async with get_file_lock(filename):
        try:
            if not path.exists():
                return
            
            loop = asyncio.get_running_loop()
            
            def _prune() -> None:
                turns = read_rec(proj)
                if not turns:
                    return
                
                now = time.time()
                cutoff = now - (FLIGHT_RECORDER_RETENTION_HOURS * 3600)
                valid_turns = [t for t in turns if t.get('ts', 0) > cutoff]
                
                if len(valid_turns) > FLIGHT_RECORDER_MAX_TURNS:
                    valid_turns = sorted(valid_turns, key=lambda x: x.get('ts', 0))[-FLIGHT_RECORDER_MAX_TURNS:]
                
                if len(valid_turns) != len(turns):
                    with open(path, "w", encoding="utf-8") as f:
                        for t in valid_turns:
                            f.write(json.dumps(t) + "\n")
                    logger.info(f"Pruned {len(turns) - len(valid_turns)} turns from {proj or 'global'}")
            
            await loop.run_in_executor(None, _prune)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def cleanup_old_turns_sync(proj: str) -> None:
    """Sync version for startup cleanup only (no async context available)."""
    p = get_rec_path(proj)
    if not p.exists():
        return
    try:
        turns = read_rec(proj)
        if not turns:
            return
        
        now = time.time()
        cutoff = now - (FLIGHT_RECORDER_RETENTION_HOURS * 3600)
        valid_turns = [t for t in turns if t.get('ts', 0) > cutoff]
        
        if len(valid_turns) > FLIGHT_RECORDER_MAX_TURNS:
            valid_turns = sorted(valid_turns, key=lambda x: x.get('ts', 0))[-FLIGHT_RECORDER_MAX_TURNS:]
            
        if len(valid_turns) != len(turns):
            with open(p, "w", encoding="utf-8") as f:
                for t in valid_turns:
                    f.write(json.dumps(t) + "\n")
            logger.info(f"Pruned {len(turns) - len(valid_turns)} turns from {proj or 'global'}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def format_recent_turns(turns_list: List[Dict[str, Any]]) -> str:
    """Format recent turns for display."""
    if not turns_list:
        return ""
    fmt: List[str] = []
    now = time.time()
    for t in turns_list:
        hours = (now - t.get('ts', 0)) / 3600
        time_str = f"{hours:.1f}h ago" if hours > 0.1 else "just now"
        user = t.get('turn', {}).get('user', '')[:MAX_TURN_CONTENT_LENGTH]
        ai = t.get('turn', {}).get('ai', '')[:MAX_TURN_CONTENT_LENGTH]
        fmt.append(f"[{time_str}]\nU: {user}\nA: {ai}")
    return "\n\n".join(fmt)


# =============================================================================
# ID GENERATION
# =============================================================================

def get_id(seed: str, salt: str = "") -> str:
    """Generate a UUID-based string ID from seed and salt."""
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{seed}_{salt}".encode().hex()))


def get_int_id(seed: str, offset: int) -> int:
    """
    Generate an integer ID from seed with offset.
    
    WARNING: This uses 8 hex characters (32 bits) with modulo 89999989,
    giving ~90 million unique values per offset range. For session and
    handover IDs this is sufficient, but be aware of collision risk
    if used for high-volume unique ID generation.
    """
    return offset + (int(hashlib.md5(seed.encode()).hexdigest()[:8], 16) % 89999989)


# =============================================================================
# EMBEDDING WITH LRU CACHE
# =============================================================================

@functools.lru_cache(maxsize=MAX_EMBEDDING_CACHE_SIZE)
def _cached_embed(text: str) -> Optional[Tuple[float, ...]]:
    """
    Generate OpenAI embedding with caching.
    Returns tuple (hashable) for LRU cache compatibility.
    """
    if not openai_client:
        logger.error("OpenAI client not initialized")
        return None
    
    try:
        response = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
        return tuple(response.data[0].embedding)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def embed(text: str) -> Optional[List[float]]:
    """Generate OpenAI embedding with caching."""
    result = _cached_embed(text)
    return list(result) if result else None


def clear_embedding_cache() -> None:
    """Clear the embedding cache (useful after API key rotation)."""
    _cached_embed.cache_clear()
    logger.info("Embedding cache cleared")


# =============================================================================
# SPARSE VECTOR GENERATION
# =============================================================================

def generate_sparse_vector(text: str) -> Dict[int, float]:
    """Generate sparse vector from text using keyword extraction."""
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    
    if not words:
        # Return a dummy vector to prevent Qdrant errors with empty payloads
        return {0: 0.001}
    
    counts = Counter(words)
    max_count = max(counts.values())
    
    vec: Dict[int, float] = {}
    for w, c in counts.items():
        idx = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % 1000000
        vec[idx] = c / max_count
    
    return vec


# =============================================================================
# POINT CREATION HELPER
# =============================================================================

def create_point(
    point_id: str,
    dense_vector: List[float],
    payload: Dict[str, Any],
    text_for_sparse: Optional[str] = None
) -> PointStruct:
    """
    Create a PointStruct with proper vector configuration.
    
    Consolidates the repeated pattern of creating points with/without
    hybrid search enabled.
    """
    if ENABLE_HYBRID_SEARCH:
        sparse_text = text_for_sparse or payload.get('content', '')
        sparse_vec = generate_sparse_vector(sparse_text)
        return PointStruct(
            id=point_id,
            vector={
                "dense": dense_vector,
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=list(sparse_vec.keys()),
                    values=list(sparse_vec.values())
                )
            },
            payload=payload
        )
    else:
        return PointStruct(id=point_id, vector=dense_vector, payload=payload)


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


# =============================================================================
# KNOWLEDGE GRAPH EXTRACTION
# =============================================================================

def extract_knowledge_graph_data(content: str) -> Optional[Dict[str, Any]]:
    """Extract entities and relationships using Gemini."""
    if not gemini_configured:
        return None
        
    prompt = f"Extract entities/relationships JSON from: {content}. Return {{'entities': [{{'name':..., 'type':...}}], 'relationships': [{{'from':..., 'to':..., 'type':...}}]}}"
    try:
        res = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config={"response_mime_type": "application/json"}
        ).generate_content(prompt)
        data = json.loads(res.text)
        return data if data.get('entities') or data.get('relationships') else None
    except Exception as e:
        logger.warning(f"Graph extraction failed: {e}")
        return None


# =============================================================================
# PROACTIVE RETRIEVAL FUNCTIONS
# =============================================================================

def detect_proactive_triggers(
    user_message: str,
    project_name: str,
    time_since_last: float
) -> List[Dict[str, Any]]:
    """Detect if proactive retrieval should fire."""
    triggers: List[Dict[str, Any]] = []
    message_lower = user_message.lower()
    
    if any(kw in message_lower for kw in EXPLICIT_MEMORY_KEYWORDS):
        triggers.append({
            "type": "explicit_memory",
            "confidence": 0.90,
            "query": user_message,
            "reason": "User explicitly referenced past memory"
        })
    
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
    
    has_project_ref = any(ref in message_lower for ref in PROJECT_CONTINUITY_KEYWORDS)
    if has_project_ref and project_name:
        triggers.append({
            "type": "project_continuity",
            "confidence": 0.70,
            "query": f"recent work {project_name}",
            "project_name": project_name,
            "reason": "Reference to ongoing project work"
        })
    
    is_greeting = any(g in message_lower for g in GREETING_KEYWORDS)
    long_gap = time_since_last > (PROACTIVE_CONFIG["min_time_gap_hours"] * 3600)
    
    if is_greeting and long_gap:
        triggers.append({
            "type": "time_based_greeting",
            "confidence": 0.65,
            "query": f"last session {project_name}" if project_name else "last session",
            "project_name": project_name,
            "reason": f"Greeting after {time_since_last/3600:.1f}h gap"
        })
    
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
    if not gemini_configured:
        return user_message
        
    prompt = f"""Given this user message: "{user_message}"
Current project: {project_name or "None"}

Generate a concise search query (2-6 words) to find the most relevant memories.
Focus on what the user is actually asking about or referencing.

Return ONLY the search query, nothing else."""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
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
    """Calculate whether a retrieved memory should be injected."""
    semantic_score = memory.get('search_score', memory.get('score', 0.5))
    decay_score = max(0.5**(
        (time.time() - memory.get('timestamp', 0)) / 
        HALF_LIVES.get(memory.get('type'), 604800)
    ), DECAY_FLOOR)
    
    age_hours = (time.time() - memory.get('timestamp', 0)) / 3600
    recency_boost = 1.5 if age_hours < 24 else (1.2 if age_hours < 72 else 1.0)
    
    type_weights: Dict[str, float] = {
        'project': 1.5, 'goal': 1.3, 'personal': 1.2, 'topic': 1.0,
        'preference': 0.9, 'dialog': 0.8, 'context_handover': 1.4
    }
    type_weight = type_weights.get(memory.get('type'), 1.0)
    boost = memory.get('boost_factor', 1.0)
    
    injection_score = (
        semantic_score * decay_score * recency_boost * 
        type_weight * boost * trigger_confidence
    )
    
    logger.debug(
        f"Injection score for '{memory.get('content', '')[:30]}...': {injection_score:.3f} "
        f"(semantic={semantic_score:.2f}, decay={decay_score:.2f}, recency={recency_boost:.1f}, "
        f"type_weight={type_weight:.1f}, trigger_conf={trigger_confidence:.2f})"
    )
    
    return injection_score


def format_proactive_context(memories: List[Dict[str, Any]]) -> str:
    """Format retrieved memories for injection."""
    if not memories:
        return ""
    
    context = "## 🧠 PROACTIVE CONTEXT\n\n"
    context += "*(Retrieved automatically based on conversation analysis)*\n\n"
    
    for i, mem in enumerate(memories[:PROACTIVE_CONFIG["max_injections"]], 1):
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


# =============================================================================
# KNOWLEDGE GRAPH BUILDER
# =============================================================================

class KnowledgeGraphBuilder:
    """Builds and queries the knowledge graph from stored memories."""
    
    def __init__(self, memory_system: 'MemorySystem'):
        self.memory_system = memory_system
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
            mems, _ = self.memory_system.client.scroll(
                COLLECTION,
                scroll_filter=filter_kg,
                limit=MAX_SCROLL_LIMIT,
                with_payload=True
            )
            for m in mems:
                d = json.loads(m.payload.get('content', '{}'))
                age = time.time() - m.payload.get('timestamp', 0)
                score = max(0.5**(age/15552000), 0.3)
                for e in d.get('entities', []):
                    if not e.get('name'):
                        continue
                    if not G.has_node(e['name']):
                        G.add_node(e['name'], type=e.get('type'), decay_score=score)
                for r in d.get('relationships', []):
                    if r.get('from') and r.get('to'):
                        G.add_edge(r['from'], r['to'], type=r.get('type'), decay_score=score)
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


# =============================================================================
# PROJECT LIST CACHE
# =============================================================================

class ProjectListCache:
    """Cache for the list of projects with TTL and invalidation."""
    
    def __init__(self, ttl: float = MAX_PROJECT_CACHE_AGE_SECONDS):
        self._cache: Optional[List[str]] = None
        self._timestamp: float = 0
        self._ttl = ttl
        self._lock = threading.Lock()
    
    def get(self) -> Optional[List[str]]:
        """Get cached project list if still valid."""
        with self._lock:
            if self._cache and (time.time() - self._timestamp) < self._ttl:
                return self._cache
            return None
    
    def set(self, projects: List[str]) -> None:
        """Update the cached project list."""
        with self._lock:
            self._cache = projects
            self._timestamp = time.time()
    
    def invalidate(self) -> None:
        """Force cache refresh on next access."""
        with self._lock:
            self._cache = None
            self._timestamp = 0


_project_cache = ProjectListCache()


# =============================================================================
# CORE MEMORY SYSTEM CLASS
# =============================================================================

class MemorySystem:
    """Core memory system with vector search, knowledge graph, and session management."""
    
    def __init__(self):
        self.client = OfficialQdrantClient(host="localhost", port=6333)
        self.init_db()
        self.graph_builder = KnowledgeGraphBuilder(self)
        logger.info("Loading Decay Memory Server V10.11 (Concurrency Fixes)")
        self.last_interaction_time: float = time.time()
        
        # Create flight recorder directory
        FLIGHT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Cleanup old turns on startup (sync - no event loop yet)
        if FLIGHT_DIR.exists():
            for f in FLIGHT_DIR.glob("flight_recorder_*.jsonl"):
                cleanup_old_turns_sync(f.stem.replace("flight_recorder_", ""))
    
    def init_db(self) -> None:
        """Initialize or verify the Qdrant collection."""
        try:
            self.client.get_collection(COLLECTION)
            logger.info("Collection exists (V10.11 Hybrid)")
        except Exception:
            logger.info("Creating new collection (Hybrid)")
            vectors_config = {
                "dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
            }
            
            if ENABLE_HYBRID_SEARCH:
                sparse_config = {
                    SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams())
                }
                self.client.create_collection(
                    COLLECTION,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_config,
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(type="int8", always_ram=True)
                    )
                )
            else:
                self.client.recreate_collection(
                    COLLECTION,
                    vectors_config=vectors_config["dense"]
                )
            
            # Create payload indexes
            self.client.create_payload_index(COLLECTION, "project_name", "keyword")
            self.client.create_payload_index(COLLECTION, "type", "keyword")
            self.client.create_payload_index(COLLECTION, "timestamp", "float")
            self.client.create_payload_index(COLLECTION, "chat_id", "keyword")
        
        self.reset_session(None, init_only=True)
    
    def reset_session(self, proj: Optional[str], init_only: bool = False) -> None:
        """Reset or initialize session state."""
        sid = get_int_id(f"session_{proj}", 10)
        if init_only and self.client.retrieve(COLLECTION, ids=[sid]):
            return
        
        payload = {
            "type": "session_state",
            "turn_count": 0,
            "tokens_used": 0,
            "session_start": time.time(),
            "last_update": time.time(),
            "session_id": str(uuid.uuid4()),
            "project_name": proj or ""
        }
        
        point = create_empty_point(sid, payload)
        self.client.upsert(COLLECTION, points=[point])
    
    def update_session(self, msg: str, proj: Optional[str]) -> Dict[str, Any]:
        """Update session turn count and metadata."""
        sid = get_int_id(f"session_{proj}", 10)
        res = self.client.retrieve(COLLECTION, ids=[sid])
        if not res:
            self.reset_session(proj)
            res = self.client.retrieve(COLLECTION, ids=[sid])
        
        st = res[0].payload
        st['turn_count'] += 1
        st['last_update'] = time.time()
        
        point = create_empty_point(sid, st)
        self.client.upsert(COLLECTION, points=[point])
        return {"state": st}
    
    def get_project_filter(
        self,
        proj: Optional[str],
        include_global: bool = True
    ) -> Optional[Filter]:
        """
        Build a Qdrant filter for project-scoped queries.
        
        V10.11: This now builds combined project+global filters for single-query approach.
        """
        if proj == "*":
            return None
        
        if not proj:
            # Global only
            return Filter(should=[
                FieldCondition(key="project_name", match=MatchValue(value="")),
                IsNullCondition(is_null=PayloadField(key="project_name"))
            ])
            
        if include_global:
            # Project + Global in single query
            return Filter(should=[
                FieldCondition(key="project_name", match=MatchValue(value=proj)),
                FieldCondition(key="project_name", match=MatchValue(value="")),
                IsNullCondition(is_null=PayloadField(key="project_name"))
            ])
        
        # Project only
        return Filter(must=[
            FieldCondition(key="project_name", match=MatchValue(value=proj))
        ])
    
    def check_conflict(
        self,
        content: str,
        mtype: str,
        proj: str
    ) -> Optional[Dict[str, Any]]:
        """Check for conflicting memories of the same type."""
        if mtype not in CONFLICT_TYPES:
            return None
        
        vec = embed(content)
        if vec is None:
            logger.warning("Failed to generate embedding for conflict check")
            return None
        
        # Build filter: must match type AND (project OR global)
        type_condition = FieldCondition(key="type", match=MatchValue(value=mtype))
        
        if proj and proj != "*":
            # Project-specific: include project memories and global
            combined_filter = Filter(
                must=[type_condition],
                should=[
                    FieldCondition(key="project_name", match=MatchValue(value=proj)),
                    FieldCondition(key="project_name", match=MatchValue(value="")),
                    IsNullCondition(is_null=PayloadField(key="project_name"))
                ]
            )
        else:
            # Global only
            combined_filter = Filter(
                must=[type_condition],
                should=[
                    FieldCondition(key="project_name", match=MatchValue(value="")),
                    IsNullCondition(is_null=PayloadField(key="project_name"))
                ]
            )
        
        if ENABLE_HYBRID_SEARCH:
            hits = self.client.search(
                COLLECTION, 
                query_vector=NamedVector(name="dense", vector=vec),
                query_filter=combined_filter, 
                limit=5, 
                with_payload=True
            )
        else:
            hits = self.client.search(
                COLLECTION,
                query_vector=vec,
                query_filter=combined_filter,
                limit=5,
                with_payload=True
            )
        
        for h in hits:
            if h.score >= CONFLICT_SIMILARITY_THRESHOLD:
                return h.payload | {'id': h.id}
        return None
    
    def resolve_conflict(
        self,
        new_content: str,
        old_memory: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to determine relationship between conflicting memories."""
        if not gemini_configured:
            return None
            
        prompt = (
            f'Old: "{old_memory.get("content")}"\n'
            f'New: "{new_content}"\n'
            f'Rel: 1.SUPERSEDE 2.UPDATE 3.COMPLEMENT(merge) 4.UNRELATED\n'
            f'JSON: {{ "rel": "supersede|update|complement|unrelated", "merged": "string", "conf": float }}'
        )
        try:
            res = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                generation_config={"response_mime_type": "application/json"}
            ).generate_content(prompt)
            return json.loads(res.text)
        except Exception as e:
            logger.warning(f"Error resolving conflict: {e}")
            return None
    
    def replace_memory(
        self,
        old_id: str,
        new_content: str,
        old_mem: Dict[str, Any]
    ) -> Optional[str]:
        """Replace an old memory with new content."""
        dense_vec = embed(new_content)
        if dense_vec is None:
            logger.warning("Failed to generate embedding, skipping replacement")
            return None
        
        # Archive old memory
        self.client.set_payload(
            COLLECTION,
            {"type": "archived_superseded", "superseded_by": "new"},
            points=[old_id]
        )
        
        # Create new memory
        new_mem = old_mem.copy()
        new_mem.update({
            "content": new_content,
            "timestamp": time.time(),
            "type": old_mem['type']
        })
        if 'id' in new_mem:
            del new_mem['id']
        
        pid = get_id(new_content, str(time.time()))
        point = create_point(pid, dense_vec, new_mem, new_content)
        self.client.upsert(COLLECTION, points=[point])
        
        if GRAPH_AUTO_INVALIDATE:
            self.graph_builder.invalidate_cache()
        
        _project_cache.invalidate()
        return pid
    
    def add_memory(
        self,
        content: str,
        mtype: str = "topic",
        proj: str = ""
    ) -> str:
        """Add a new memory to the system."""
        if mtype in GLOBAL_TYPES:
            proj = ""
        
        # Check for conflicts
        conflict = self.check_conflict(content, mtype, proj)
        if conflict:
            res = self.resolve_conflict(content, conflict)
            if res and res.get('conf', 0) > CONFLICT_CONFIDENCE_THRESHOLD:
                if res['rel'] == 'supersede':
                    self.replace_memory(conflict['id'], content, conflict)
                    return "Superseded old memory."
                elif res['rel'] in ['update', 'complement']:
                    self.replace_memory(conflict['id'], res['merged'], conflict)
                    return f"Merged memory: {res['merged'][:50]}..."
        
        # Generate embedding
        dense_vec = embed(content)
        if dense_vec is None:
            return "Error: Failed to generate embedding"
        
        pid = get_id(content, str(time.time()))
        payload = {
            "content": content,
            "type": mtype,
            "score": 1.0,
            "timestamp": time.time(),
            "project_name": proj,
            "access_count": 0,
            "boost_factor": 1.0,
            "deprecation_factor": 1.0,
            "is_deprecated": False,
            "boost_history": [],
            "correction_history": []
        }
        
        point = create_point(pid, dense_vec, payload, content)
        self.client.upsert(COLLECTION, points=[point])
        
        _project_cache.invalidate()
        
        # Extract knowledge graph if applicable
        if mtype in GRAPH_EXTRACTION_TYPES:
            self._extract_and_store_graph(content, proj, pid)
        
        return f"Stored: {content[:50]}..."
    
    def _extract_and_store_graph(
        self,
        content: str,
        proj: str,
        source_memory_id: str
    ) -> None:
        """Extract and store knowledge graph data from content."""
        try:
            gd = extract_knowledge_graph_data(content)
            if not gd:
                return
            
            gvec = embed(content)
            if gvec is None:
                logger.warning("Failed to generate embedding for graph entry")
                return
            
            gpid = get_id(f"graph_{content}", str(time.time()))
            gpay = {
                "content": json.dumps(gd),
                "type": "knowledge_graph",
                "score": 1.0,
                "timestamp": time.time(),
                "project_name": proj,
                "source_memory_id": source_memory_id
            }
            
            g_point = create_point(gpid, gvec, gpay, content)
            self.client.upsert(COLLECTION, points=[g_point])
            
            if GRAPH_AUTO_INVALIDATE:
                self.graph_builder.invalidate_cache()
                
        except Exception as e:
            logger.warning(f"Error creating graph entry: {e}")
    
    def hybrid_search(
        self,
        query: str,
        proj: str = "",
        limit: int = 15,
        expand: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Prefetch + Fusion (RRF).
        
        V10.11: Uses single query with combined filter (project + global)
        instead of sequential fallthrough. Halves latency.
        """
        dense_vec = embed(query)
        if dense_vec is None:
            logger.warning("Failed to generate embedding for query")
            return []
        
        # V10.11: Single filter includes both project and global
        # get_project_filter with include_global=True returns Filter(should=[project, global])
        proj_filter = self.get_project_filter(proj, include_global=True)
        results: List[Any] = []
        
        if ENABLE_HYBRID_SEARCH:
            try:
                sparse_vec = generate_sparse_vector(query)
                prefetch_limit = limit * PREFETCH_MULTIPLIER
                
                prefetch = [
                    Prefetch(
                        query=dense_vec,
                        using="dense",
                        filter=proj_filter,
                        limit=prefetch_limit
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=list(sparse_vec.keys()),
                            values=list(sparse_vec.values())
                        ),
                        using=SPARSE_VECTOR_NAME,
                        filter=proj_filter,
                        limit=prefetch_limit
                    )
                ]
                
                search_result = self.client.query_points(
                    COLLECTION,
                    prefetch=prefetch,
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=limit,
                    with_payload=True
                )
                results = search_result.points
                
            except Exception as e:
                logger.warning(f"Hybrid Search failed: {e}. Falling back to Dense.")
                results = self.client.search(
                    COLLECTION,
                    query_vector=dense_vec,
                    query_filter=proj_filter,
                    limit=limit,
                    with_payload=True
                )
        else:
            results = self.client.search(
                COLLECTION,
                query_vector=dense_vec,
                query_filter=proj_filter,
                limit=limit,
                with_payload=True
            )
        
        # Process and score results
        processed: List[Tuple[float, Dict[str, Any]]] = []
        for hit in results:
            if 'archived' in hit.payload.get('type', ''):
                continue
            if hit.payload.get('type') == 'knowledge_graph':
                continue
            
            age = time.time() - hit.payload.get('timestamp', 0)
            hl = HALF_LIVES.get(hit.payload.get('type'), 604800)
            effective_hl = hl * hit.payload.get('deprecation_factor', 1.0)
            decay = max(0.5 ** (age / effective_hl), DECAY_FLOOR)
            boost = hit.payload.get('boost_factor', 1.0)
            access_boost = 1 + hit.payload.get('access_count', 0) * 0.05
            
            final_score = hit.score * decay * boost * access_boost
            
            payload_with_score = hit.payload | {'id': hit.id, 'search_score': final_score}
            
            # V10.11: Mark source for debugging (project-specific vs global)
            result_proj = hit.payload.get('project_name', '')
            if proj and result_proj != proj:
                payload_with_score['_source'] = 'global' if not result_proj else result_proj
            
            processed.append((final_score, payload_with_score))
        
        processed.sort(key=lambda x: x[0], reverse=True)
        top_results = [p for _, p in processed]
        
        if expand and top_results:
            top_results = self._expand_with_graph(top_results, proj)
        
        return top_results[:limit]
    
    def _expand_with_graph(
        self,
        results: List[Dict[str, Any]],
        proj: str
    ) -> List[Dict[str, Any]]:
        """Expand search results using knowledge graph."""
        expanded_ids: Set[str] = {r['id'] for r in results}
        final_results = list(results)
        
        for mem in results[:MAX_ENTITIES_TO_SEARCH]:
            content = mem.get('content', '')
            entities = re.findall(r'\b[A-Z][a-z]+\b', content)
            
            for entity in entities[:MAX_GRAPH_ENTITIES_TO_EXPAND]:
                try:
                    related = self.get_related_entities(entity, depth=1)
                    if not related.get('results'):
                        continue
                    
                    for res in related['results']:
                        for rel_ent in res.get('related_entities', [])[:MAX_GRAPH_ENTITIES_TO_EXPAND]:
                            entity_name = rel_ent['entity']
                            vec = embed(entity_name)
                            if vec is None:
                                continue
                            
                            if ENABLE_HYBRID_SEARCH:
                                hits = self.client.search(
                                    COLLECTION,
                                    query_vector=NamedVector(name="dense", vector=vec),
                                    query_filter=self.get_project_filter(proj),
                                    limit=1,
                                    with_payload=True
                                )
                            else:
                                hits = self.client.search(
                                    COLLECTION,
                                    query_vector=vec,
                                    query_filter=self.get_project_filter(proj),
                                    limit=1,
                                    with_payload=True
                                )
                            
                            for h in hits:
                                if h.id not in expanded_ids and 'archived' not in h.payload.get('type', ''):
                                    final_results.append(h.payload | {'id': h.id})
                                    expanded_ids.add(h.id)
                except Exception:
                    continue
        
        return final_results
    
    def search(
        self,
        query: str,
        proj: str = "",
        limit: int = 5,
        return_format: str = "compact"
    ) -> List[Dict[str, Any]]:
        """Main search entry point - uses hybrid search if enabled."""
        results = self.hybrid_search(query, proj, limit=limit, expand=True)
        
        if return_format == "full":
            return results
        
        # Compact format
        compact_results: List[Dict[str, Any]] = []
        for mem in results:
            compact = {
                "content": condense_text(mem.get("content", ""), aggressive=False),
                "type": mem.get("type", "unknown"),
                "project_name": mem.get("project_name"),
                "id": mem.get("id")
            }
            
            # V10.11: Include source info if from different project/global
            if mem.get('_source'):
                compact['_source'] = mem.get('_source')
                
            compact_results.append(compact)
            
        return compact_results
    
    # --- FEEDBACK METHODS ---
    
    def boost_memory(self, memory_id: str, reason: str = "") -> str:
        """Increase memory importance."""
        try:
            res = self.client.retrieve(COLLECTION, ids=[memory_id], with_payload=True)
            if not res:
                return "❌ Memory not found"
            mem = res[0]
            curr = mem.payload.get('boost_factor', 1.0)
            new_b = min(curr * BOOST_INCREMENT, MAX_BOOST)
            hist = mem.payload.get('boost_history', [])
            hist.append({"ts": time.time(), "reason": reason, "old": curr, "new": new_b})
            self.client.set_payload(
                COLLECTION,
                {"boost_factor": new_b, "boost_history": hist},
                points=[memory_id]
            )
            return f"✅ Boosted {curr:.1f}x -> {new_b:.1f}x"
        except Exception as e:
            return f"❌ Error: {e}"
    
    def deprecate_memory(self, memory_id: str, reason: str = "") -> str:
        """Mark memory as deprecated."""
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
            return "✅ Memory deprecated."
        except Exception as e:
            return f"❌ Error: {e}"
    
    def correct_memory(self, memory_id: str, new_content: str) -> str:
        """Correct memory content."""
        try:
            res = self.client.retrieve(COLLECTION, ids=[memory_id], with_payload=True)
            if not res:
                return "❌ Memory not found"
            mem = res[0]
            hist = mem.payload.get('correction_history', [])
            hist.append({
                "ts": time.time(),
                "old": mem.payload.get('content'),
                "new": new_content
            })
            
            new_vec = embed(new_content)
            if new_vec is None:
                return "❌ Error: Failed to generate embedding"
            
            updated = mem.payload.copy()
            updated.update({
                "content": new_content,
                "timestamp": time.time(),
                "correction_history": hist
            })
            
            point = create_point(memory_id, new_vec, updated, new_content)
            self.client.upsert(COLLECTION, points=[point])
            return "✅ Memory corrected."
        except Exception as e:
            return f"❌ Error: {e}"
    
    def delete_memory(self, memory_id: str, reason: str = "") -> str:
        """Archive/delete a memory."""
        try:
            self.client.set_payload(
                COLLECTION,
                {"type": "archived_deleted", "del_reason": reason},
                points=[memory_id]
            )
            _project_cache.invalidate()
            return "✅ Memory deleted."
        except Exception as e:
            return f"❌ Error: {e}"
    
    def set_memory_project(self, memory_id: str, project_name: str) -> str:
        """Update a memory's project assignment."""
        try:
            res = self.client.retrieve(COLLECTION, ids=[memory_id], with_payload=True)
            if not res:
                return "❌ Memory not found"
            
            old_project = res[0].payload.get('project_name', '')
            self.client.set_payload(
                COLLECTION,
                {"project_name": project_name},
                points=[memory_id]
            )
            
            _project_cache.invalidate()
            return f"✅ Moved: '{old_project or 'global'}' → '{project_name or 'global'}'"
        except Exception as e:
            return f"❌ Error: {e}"
    
    def set_chat_project(self, chat_id: str, project_name: str) -> str:
        """Update project for ALL memories from a specific chat."""
        try:
            chat_filter = Filter(must=[
                FieldCondition(key="chat_id", match=MatchValue(value=chat_id))
            ])
            results, _ = self.client.scroll(
                COLLECTION,
                scroll_filter=chat_filter,
                limit=MAX_SCROLL_LIMIT,
                with_payload=True
            )
            
            if not results:
                return f"❌ No memories found with chat_id: {chat_id}"
            
            chat_title = results[0].payload.get('chat_title', 'Unknown')
            old_project = results[0].payload.get('project_name', '')
            
            point_ids = [r.id for r in results]
            self.client.set_payload(
                COLLECTION,
                {"project_name": project_name},
                points=point_ids
            )
            
            _project_cache.invalidate()
            return (
                f"✅ Updated {len(point_ids)} memories from '{chat_title}': "
                f"'{old_project or 'global'}' → '{project_name or 'global'}'"
            )
        except Exception as e:
            return f"❌ Error: {e}"
    
    def list_projects(self) -> List[str]:
        """Get list of all projects (cached)."""
        cached = _project_cache.get()
        if cached is not None:
            return cached
        
        res, _ = self.client.scroll(COLLECTION, limit=MAX_SCROLL_LIMIT)
        projects = sorted(list({
            p.payload.get('project_name')
            for p in res
            if p.payload.get('project_name')
        }))
        
        _project_cache.set(projects)
        return projects
    
    # --- GRAPH WRAPPERS ---
    
    def find_connection(
        self,
        entity_a: str,
        entity_b: str,
        max_hops: int = 3
    ) -> Dict[str, Any]:
        """Find connection between two entities."""
        return self.graph_builder.find_connection(entity_a, entity_b, max_hops)
    
    def get_related_entities(
        self,
        entity: str,
        relationship_type: Optional[str] = None,
        depth: int = 1
    ) -> Dict[str, Any]:
        """Get entities related to the given entity."""
        return self.graph_builder.get_related_entities(entity, relationship_type, depth)
    
    def get_entity_neighborhood(
        self,
        entity: str,
        radius: int = 2
    ) -> Dict[str, Any]:
        """Get the subgraph around an entity."""
        return self.graph_builder.get_entity_neighborhood(entity, radius)
    
    # --- PROACTIVE RETRIEVAL ---
    
    def proactive_retrieval(
        self,
        user_message: str,
        project_name: str
    ) -> Optional[str]:
        """Proactively retrieve relevant memories based on user message."""
        if not PROACTIVE_CONFIG["enabled"]:
            return None
        
        cache_key = f"{user_message[:50]}_{project_name}"
        cached_result = _proactive_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Proactive cache hit for: {cache_key[:30]}")
            return cached_result
        
        time_gap = time.time() - getattr(self, 'last_interaction_time', 0)
        triggers = detect_proactive_triggers(user_message, project_name, time_gap)
        
        if not triggers:
            logger.debug("Proactive: No triggers detected")
            self.last_interaction_time = time.time()
            return None
        
        valid_triggers = [
            t for t in triggers
            if t['confidence'] >= PROACTIVE_CONFIG['trigger_threshold']
        ]
        
        if not valid_triggers:
            logger.debug(
                f"Proactive: {len(triggers)} triggers below threshold "
                f"{PROACTIVE_CONFIG['trigger_threshold']}"
            )
            self.last_interaction_time = time.time()
            return None
        
        logger.info(
            f"Proactive fired {len(valid_triggers)} triggers: "
            f"{[t['type'] for t in valid_triggers]}"
        )
        
        all_memories: List[Dict[str, Any]] = []
        fast_path_success = False
        
        for trigger in valid_triggers:
            if trigger['type'] in ['explicit_memory', 'entity_mention']:
                query = trigger['query']
                memories = self.search(
                    query,
                    trigger.get('project_name', project_name),
                    return_format="full"
                )
                
                for mem in memories:
                    mem['injection_score'] = calculate_injection_score(
                        mem, user_message, trigger['confidence']
                    )
                
                all_memories.extend(memories)
                
                if memories:
                    max_score = max(m['injection_score'] for m in memories)
                    if max_score >= PROACTIVE_CONFIG['fast_path_threshold']:
                        fast_path_success = True
        
        if not fast_path_success and PROACTIVE_CONFIG['use_llm_expansion']:
            llm_triggers = [
                t for t in valid_triggers
                if t.get('needs_llm', False) or
                t['type'] in ['project_continuity', 'question_pattern', 'time_based_greeting']
            ]
            
            for trigger in llm_triggers:
                better_query = generate_llm_query(user_message, project_name)
                logger.debug(f"Proactive LLM query: '{better_query}'")
                
                memories = self.search(
                    better_query,
                    trigger.get('project_name', project_name),
                    return_format="full"
                )
                
                for mem in memories:
                    mem['injection_score'] = calculate_injection_score(
                        mem, user_message, trigger['confidence']
                    )
                
                all_memories.extend(memories)
        
        if PROACTIVE_CONFIG['include_graph']:
            entity_triggers = [t for t in valid_triggers if t.get('graph_expand', False)]
            
            for trigger in entity_triggers:
                entity = trigger['query']
                try:
                    related = self.get_related_entities(entity, depth=1)
                    
                    if 'results' in related and related['results']:
                        for result in related['results']:
                            for rel_entity in result.get('related_entities', [])[:MAX_RELATED_ENTITIES_PER_RESULT]:
                                entity_name = rel_entity['entity']
                                entity_mems = self.search(
                                    entity_name,
                                    project_name,
                                    return_format="full"
                                )
                                
                                for mem in entity_mems:
                                    mem['injection_score'] = calculate_injection_score(
                                        mem, user_message, trigger['confidence'] * 0.8
                                    )
                                
                                all_memories.extend(entity_mems)
                except Exception:
                    pass
        
        # Deduplicate
        seen_ids: Set[str] = set()
        unique_mems: List[Dict[str, Any]] = []
        for mem in all_memories:
            mem_id = mem.get('id')
            if mem_id and mem_id not in seen_ids:
                seen_ids.add(mem_id)
                unique_mems.append(mem)
        
        unique_mems.sort(key=lambda m: m.get('injection_score', 0), reverse=True)
        
        injection_threshold = PROACTIVE_CONFIG['injection_threshold']
        top_memories = [
            m for m in unique_mems
            if m.get('injection_score', 0) >= injection_threshold
        ]
        
        if unique_mems and not top_memories:
            best_score = max(m.get('injection_score', 0) for m in unique_mems)
            logger.debug(
                f"Proactive: Best injection score {best_score:.3f} < threshold {injection_threshold}"
            )
        
        result: Optional[str] = None
        if top_memories:
            result = format_proactive_context(top_memories)
            logger.info(f"Proactive injecting {len(top_memories)} memories")
        
        _proactive_cache.set(cache_key, result)
        
        self.last_interaction_time = time.time()
        return result
    
    def update_proactive_config(self, updates: Dict[str, Any]) -> str:
        """Update proactive retrieval configuration."""
        valid_keys = PROACTIVE_CONFIG.keys()
        updated: List[str] = []
        
        for key, value in updates.items():
            if key in valid_keys:
                PROACTIVE_CONFIG[key] = value
                updated.append(key)
        
        return f"✅ Updated config: {', '.join(updated)}"
    
    # --- SESSION FINALIZATION (V10.11: Now async to respect file locks) ---
    
    async def finalize_async(self, summary: str, proj: str) -> str:
        """
        Finalize session with fallback for JSON parsing failures.
        
        V10.11: Now async to properly use cleanup_old_turns_async and respect
        the same asyncio.Lock as log_turn_async. Prevents race conditions
        where finalize could overwrite concurrent log writes.
        """
        if not gemini_configured:
            logger.warning("Gemini not configured, skipping LLM summarization")
            await cleanup_old_turns_async(proj)
            self.reset_session(proj)
            return "✅ Session Finalized (without LLM summary)."
        
        # Try to extract structured facts
        prompt = f"Extract facts JSON list. Types: personal,project,topic.\n{summary}"
        try:
            result = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                generation_config={"response_mime_type": "application/json"}
            ).generate_content(prompt)
            mems = json.loads(result.text)
            for m in mems:
                self.add_memory(m['content'], m['type'], proj)
        except Exception as e:
            logger.warning(f"Finalize extraction failed: {e}. Storing raw summary.")
            self.add_memory(
                f"Session Summary (Raw): {summary[:MAX_RAW_SUMMARY_LENGTH]}",
                "dialog",
                proj
            )
        
        # Create handover
        try:
            prompt = (
                f"Summarize for next session. Project: {proj}. "
                f"JSON: summary, last_topic, unresolved.\n{summary}"
            )
            result = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                generation_config={"response_mime_type": "application/json"}
            ).generate_content(prompt)
            ho = json.loads(result.text)
            hid = get_int_id(f"handover_{proj}", 100000000)
            vec = embed(ho['summary'])
            
            if vec is None:
                logger.warning("Failed to generate embedding for handover")
            else:
                payload = {
                    "content": ho['summary'],
                    "type": "context_handover",
                    "project_name": proj or "",
                    "timestamp": time.time(),
                    **ho
                }
                
                point = create_point(hid, vec, payload, ho['summary'])
                self.client.upsert(COLLECTION, points=[point])
        except Exception as e:
            logger.warning(f"Error creating handover: {e}")
        
        # V10.11: Use async cleanup to respect file locks
        await cleanup_old_turns_async(proj)
        self.reset_session(proj)
        return "✅ Session Finalized. Flight recorder pruned."
    
    def get_handover(
        self,
        proj: str,
        format: str = "condensed",
        max_turns: int = 20
    ) -> str:
        """Get context for a project with token optimization."""
        output: List[str] = []
        
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

        turns = read_rec(proj)
        
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
            else:
                context_str = self._format_turns_condensed(recent_turns)
                
            output.append(
                f"📝 RECENT CONTEXT ({format}, {len(recent_turns)} turns):\n{context_str}"
            )
        
        hid = get_int_id(f"handover_{proj}", 100000000)
        res = self.client.retrieve(COLLECTION, ids=[hid], with_payload=True)
        if res and res[0].payload:
            summary_text = res[0].payload['content']
            if format != "full":
                summary_text = condense_text(summary_text)
            output.append(f"📚 OLDER SUMMARY:\n{summary_text}")
            
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


# =============================================================================
# MCP SERVER
# =============================================================================

app = Server("decay-memory")
sys_core: Optional[MemorySystem] = None


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="add_memory",
            description="Store memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "memory_type": {"type": "string"},
                    "project_name": {"type": "string"}
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="search_memories",
            description="Search memories (Compact by default).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "project_name": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "return_format": {
                        "type": "string",
                        "enum": ["compact", "full"],
                        "default": "compact"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_memories_with_ids",
            description="Search with visible IDs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "project_name": {"type": "string"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="finalize_session",
            description="End session & save.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_summary": {"type": "string"},
                    "project_name": {"type": "string"}
                },
                "required": ["conversation_summary"]
            }
        ),
        Tool(
            name="update_session_state",
            description="Log turn.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message_text": {"type": "string"},
                    "project_name": {"type": "string"}
                },
                "required": ["message_text"]
            }
        ),
        Tool(
            name="update_last_turn",
            description="Update flight recorder with AI response summary (Phase 2 of two-phase logging).",
            inputSchema={
                "type": "object",
                "properties": {
                    "ai_response_summary": {
                        "type": "string",
                        "description": "Brief 1-2 sentence summary of AI's response"
                    },
                    "project_name": {"type": "string"}
                },
                "required": ["ai_response_summary"]
            }
        ),
        Tool(
            name="get_context_handover",
            description="Get context (Condensed by default).",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["full", "condensed", "minimal"],
                        "default": "condensed"
                    },
                    "max_turns": {"type": "integer", "default": 20}
                }
            }
        ),
        Tool(
            name="list_projects",
            description="List projects.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="find_connection",
            description="Find graph path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_a": {"type": "string"},
                    "entity_b": {"type": "string"}
                },
                "required": ["entity_a", "entity_b"]
            }
        ),
        Tool(
            name="get_related_entities",
            description="Get neighbors.",
            inputSchema={
                "type": "object",
                "properties": {"entity": {"type": "string"}},
                "required": ["entity"]
            }
        ),
        Tool(
            name="get_entity_neighborhood",
            description="Get subgraph.",
            inputSchema={
                "type": "object",
                "properties": {"entity": {"type": "string"}},
                "required": ["entity"]
            }
        ),
        Tool(
            name="boost_memory",
            description="Boost importance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="deprecate_memory",
            description="Mark outdated.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="correct_memory",
            description="Fix content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "new_content": {"type": "string"}
                },
                "required": ["memory_id", "new_content"]
            }
        ),
        Tool(
            name="delete_memory",
            description="Archive memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="set_memory_project",
            description="Move individual memory to different project.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "project_name": {"type": "string"}
                },
                "required": ["memory_id", "project_name"]
            }
        ),
        Tool(
            name="set_chat_project",
            description="Move ALL memories from a chat to a project.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"},
                    "project_name": {"type": "string"}
                },
                "required": ["chat_id", "project_name"]
            }
        ),
        Tool(
            name="proactive_retrieve",
            description="Proactively retrieve relevant memories based on user message. Returns formatted context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_message": {"type": "string"},
                    "project_name": {"type": "string"}
                },
                "required": ["user_message"]
            }
        ),
        Tool(
            name="configure_proactive",
            description="Update proactive retrieval settings.",
            inputSchema={
                "type": "object",
                "properties": {"config": {"type": "object"}},
                "required": ["config"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, args: Any) -> list[TextContent]:
    """Handle MCP tool calls."""
    if sys_core is None:
        return [TextContent(type="text", text=json.dumps({"error": "System not initialized"}))]
    
    if name == "add_memory":
        result = await asyncio.to_thread(
            sys_core.add_memory,
            args.get("content"),
            args.get("memory_type"),
            args.get("project_name")
        )
        return [TextContent(type="text", text=json.dumps({"result": result}, default=str))]
    
    elif name == "search_memories":
        result = await asyncio.to_thread(
            sys_core.search,
            args.get("query"),
            args.get("project_name"),
            limit=args.get("limit", 5),
            return_format=args.get("return_format", "compact")
        )
        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
    
    elif name == "search_memories_with_ids":
        result = await asyncio.to_thread(
            sys_core.search,
            args.get("query"),
            args.get("project_name")
        )
        fmt = ["=== RESULTS WITH IDs ==="] + [
            f"\nID: {r['id']}\nContent: {r['content'][:100]}...\nBoost: {r.get('boost_factor', 1.0):.1f}x"
            for r in result
        ]
        return [TextContent(type="text", text="\n".join(fmt))]
    
    elif name == "finalize_session":
        # V10.11: Call async method directly (no to_thread)
        result = await sys_core.finalize_async(
            args.get("conversation_summary"),
            args.get("project_name")
        )
        return [TextContent(type="text", text=json.dumps({"result": result}, default=str))]
    
    elif name == "update_session_state":
        await log_turn_async(
            args.get("project_name"),
            args.get("message_text"),
            "(AI Response Pending)"
        )
        result = await asyncio.to_thread(
            sys_core.update_session,
            args.get("message_text"),
            args.get("project_name")
        )
        return [TextContent(type="text", text=json.dumps(result, default=str))]
    
    elif name == "update_last_turn":
        success = await update_last_turn_response_async(
            args.get("project_name"),
            args.get("ai_response_summary")
        )
        return [TextContent(type="text", text=json.dumps({"success": success}, default=str))]
    
    elif name == "get_context_handover":
        result = await asyncio.to_thread(
            sys_core.get_handover,
            args.get("project_name"),
            format=args.get("format", "condensed"),
            max_turns=args.get("max_turns", 20)
        )
        return [TextContent(type="text", text=result)]
    
    elif name == "list_projects":
        result = await asyncio.to_thread(sys_core.list_projects)
        return [TextContent(type="text", text=json.dumps(result, default=str))]
    
    elif name == "find_connection":
        result = await asyncio.to_thread(
            sys_core.find_connection,
            args.get("entity_a"),
            args.get("entity_b")
        )
        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
    
    elif name == "get_related_entities":
        result = await asyncio.to_thread(
            sys_core.get_related_entities,
            args.get("entity"),
            args.get("relationship_type")
        )
        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
    
    elif name == "get_entity_neighborhood":
        result = await asyncio.to_thread(
            sys_core.get_entity_neighborhood,
            args.get("entity")
        )
        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
    
    elif name == "boost_memory":
        result = await asyncio.to_thread(
            sys_core.boost_memory,
            args.get("memory_id"),
            args.get("reason")
        )
        return [TextContent(type="text", text=json.dumps({"result": result}, default=str))]
    
    elif name == "deprecate_memory":
        result = await asyncio.to_thread(
            sys_core.deprecate_memory,
            args.get("memory_id"),
            args.get("reason")
        )
        return [TextContent(type="text", text=json.dumps({"result": result}, default=str))]
    
    elif name == "correct_memory":
        result = await asyncio.to_thread(
            sys_core.correct_memory,
            args.get("memory_id"),
            args.get("new_content")
        )
        return [TextContent(type="text", text=json.dumps({"result": result}, default=str))]
    
    elif name == "delete_memory":
        result = await asyncio.to_thread(
            sys_core.delete_memory,
            args.get("memory_id"),
            args.get("reason")
        )
        return [TextContent(type="text", text=json.dumps({"result": result}, default=str))]
    
    elif name == "set_memory_project":
        result = await asyncio.to_thread(
            sys_core.set_memory_project,
            args.get("memory_id"),
            args.get("project_name")
        )
        return [TextContent(type="text", text=json.dumps({"result": result}, default=str))]
    
    elif name == "set_chat_project":
        result = await asyncio.to_thread(
            sys_core.set_chat_project,
            args.get("chat_id"),
            args.get("project_name")
        )
        return [TextContent(type="text", text=json.dumps({"result": result}, default=str))]
    
    elif name == "proactive_retrieve":
        context = await asyncio.to_thread(
            sys_core.proactive_retrieval,
            args.get("user_message"),
            args.get("project_name", "")
        )
        return [TextContent(type="text", text=context or "No proactive context")]
    
    elif name == "configure_proactive":
        result = await asyncio.to_thread(
            sys_core.update_proactive_config,
            args.get("config", {})
        )
        return [TextContent(type="text", text=json.dumps({"result": result}, default=str))]
    
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main() -> None:
    """Main entry point with proper API key validation."""
    global sys_core, openai_client, gemini_configured
    
    # Validate API keys
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment!")
        logger.info("Set it in .env file or system environment variables")
        sys.exit(1)
    
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found in environment!")
        logger.info("LLM features (summarization, graph extraction) will be disabled.")
    
    # Initialize clients
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
    
    logger.info("OpenAI client initialized")
    if gemini_configured:
        logger.info("Gemini client initialized")
    
    # Initialize memory system
    sys_core = MemorySystem()
    
    # Start MCP server
    from mcp.server.stdio import stdio_server
    logger.info("Server ready. Listening on stdio...")
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
