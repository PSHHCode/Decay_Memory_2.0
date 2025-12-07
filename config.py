"""
Centralized Configuration for Decay Memory App
All sensitive values loaded from environment variables
Hot-reload support for runtime tuning parameters
"""
import os
import json
import threading
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("DecayMemory")

# =============================================================================
# HOT-RELOAD CONFIGURATION SYSTEM (Restored from v1.0)
# =============================================================================

CONFIG_FILE = Path(__file__).parent / "dashboard_config.json"
_config_lock = threading.Lock()
CONFIG_CACHE: Dict[str, Any] = {"data": None, "mtime": 0}


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


def set_config_value(path: str, value: Any) -> bool:
    """Set a config value by path and save to file."""
    global CONFIG_CACHE
    
    with _config_lock:
        try:
            config = load_config() or {}
            
            keys = path.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            
            CONFIG_CACHE["data"] = config
            CONFIG_CACHE["mtime"] = CONFIG_FILE.stat().st_mtime
            logger.info(f"Updated config: {path} = {value}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False


def get_full_config() -> Dict[str, Any]:
    """Get the full configuration dictionary."""
    return load_config() or {}


def reload_config() -> bool:
    """Force reload of configuration from disk."""
    global CONFIG_CACHE
    with _config_lock:
        CONFIG_CACHE["mtime"] = 0  # Invalidate cache
    load_config()
    return True

# --- API Keys (from environment only) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# --- Service URLs ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", f"http://{QDRANT_HOST}:{QDRANT_PORT}")

# --- Collection Settings ---
COLLECTION_NAME = "decay_memory_mcp"
VECTOR_DIMENSION = 1536
SPARSE_VECTOR_NAME = "keywords"
ENABLE_HYBRID_SEARCH = True

# --- Embedding Settings ---
EMBED_MODEL = "text-embedding-3-small"
MAX_EMBEDDING_CACHE_SIZE = 1000

# --- Memory Decay Settings ---
DECAY_RATE = 0.995
DECAY_FLOOR = 0.1

# --- API Settings ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- Validation ---
def validate_config():
    """Validate required configuration is present"""
    required = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
