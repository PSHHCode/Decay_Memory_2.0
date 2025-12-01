import os
import functools
import logging
from typing import Optional, List, Tuple

from openai import OpenAI

# --- CONFIGURATION ---
MAX_EMBEDDING_CACHE_SIZE = 1000
EMBED_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger("EmbeddingHelper")

# Initialize OpenAI client
openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.error("OPENAI_API_KEY not found in environment!")

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
