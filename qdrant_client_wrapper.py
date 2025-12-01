import logging
from qdrant_client import QdrantClient as OfficialQdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, ScalarQuantization, ScalarQuantizationConfig,
    SparseVectorParams, SparseIndexParams
)

# --- CONFIGURATION ---
COLLECTION = "decay_memory_mcp"
EMBED_DIM = 1536
SPARSE_VECTOR_NAME = "keywords"
ENABLE_HYBRID_SEARCH = True

logger = logging.getLogger("QdrantWrapper")

import os
from qdrant_client import QdrantClient

def get_qdrant_client():
    # Use 'qdrant' as default if running in Docker, else 'localhost'
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", 6333))
    
    print(f"🔌 Connecting to Qdrant at {host}:{port}...")
    return QdrantClient(host=host, port=port)

def init_db(client: OfficialQdrantClient):
    """Initialize or verify the Qdrant collection."""
    try:
        client.get_collection(COLLECTION)
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
            client.create_collection(
                COLLECTION,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_config,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(type="int8", always_ram=True)
                )
            )
        else:
            client.recreate_collection(
                COLLECTION,
                vectors_config=vectors_config["dense"]
            )
        
        # Create payload indexes
        client.create_payload_index(COLLECTION, "project_name", "keyword")
        client.create_payload_index(COLLECTION, "type", "keyword")
        client.create_payload_index(COLLECTION, "user_id", "keyword")
        client.create_payload_index(COLLECTION, "timestamp", "float")
        client.create_payload_index(COLLECTION, "chat_id", "keyword")
