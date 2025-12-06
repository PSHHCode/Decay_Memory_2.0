"""
Centralized Configuration for Decay Memory App
All sensitive values loaded from environment variables
"""
import os
from dotenv import load_dotenv

load_dotenv()

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
