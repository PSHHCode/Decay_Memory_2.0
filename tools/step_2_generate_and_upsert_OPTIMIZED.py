import os
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from openai import OpenAI

load_dotenv()

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "decay_memory_mcp"
CONTENT_FIELD = "content"
BATCH_SIZE = 64  # Smaller batch size is safer for large embedding models

# CRITICAL FIX 1: Match the model from your MCP server script
MODEL_NAME = "text-embedding-3-small"
EXPECTED_DIM = 1536  # This model is 1536 dimensions

# --- Initialize Client and Model ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print(f"## 2. Initializing Embedding Model: {MODEL_NAME}...")

# Use the OpenAI client, matching your server script
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not set in environment")
    exit(1)

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=20.0)
    print("OpenAI Client initialized.")
except Exception as e:
    print(f"❌ ERROR: Failed to initialize OpenAI client.")
    print(f"Details: {e}")
    exit()


def get_embeddings_batch(texts: list) -> list:
    """Helper function to get embeddings from OpenAI."""
    try:
        response = openai_client.embeddings.create(
            model=MODEL_NAME,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"    ✗ ERROR generating embeddings: {e}")
        return [None] * len(texts)


def has_embedding(vector) -> bool:
    """Check if a point already has a valid embedding (non-zero vector)
    Handles both named vectors (dict) and unnamed vectors (list)
    """
    if vector is None:
        return False
    
    # Handle named vectors (dict like {"dense": [...], "keywords": ...})
    if isinstance(vector, dict):
        dense = vector.get('dense')
        if dense is None:
            return False
        return not all(v == 0.0 for v in dense)
    
    # Handle unnamed vectors (list)
    return not all(v == 0.0 for v in vector)


def generate_and_upsert_vectors():
    print(f"\n## 3. Generating and Upserting Vectors for '{COLLECTION_NAME}'...")
    
    # 1. Fetch all points and filter for those WITHOUT embeddings
    points_to_process = []
    points_with_embeddings = 0
    next_offset = None
    
    print("🔍 Scanning collection for points without embeddings...")
    
    while True:
        scroll_result, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None, 
            limit=200,
            with_payload=True,
            with_vectors=True,  # Need to check vectors
            offset=next_offset
        )
        
        # Filter out points that already have embeddings
        for point in scroll_result:
            if has_embedding(point.vector):
                points_with_embeddings += 1
            else:
                points_to_process.append(point)
        
        if next_offset is None:
            break

    print(f"✅ Scan complete:")
    print(f"   - Points with embeddings (skipping): {points_with_embeddings}")
    print(f"   - Points needing embeddings: {len(points_to_process)}")

    if not points_to_process:
        print("\n🎉 All points already have embeddings! Nothing to do.")
        return

    # 2. Process and Upsert in Batches
    total_processed = 0
    total_batches = (len(points_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(points_to_process), BATCH_SIZE):
        batch_points = points_to_process[i:i + BATCH_SIZE]
        
        # Extract texts from the payload
        texts_to_embed = [p.payload[CONTENT_FIELD] for p in batch_points if p.payload and CONTENT_FIELD in p.payload]
        
        if not texts_to_embed:
            continue
            
        batch_num = i//BATCH_SIZE + 1
        print(f"--- Processing batch {batch_num} of {total_batches} ({len(texts_to_embed)} points) ---")
        
        # Generate embeddings for the batch
        print("    🧠 Generating embeddings...")
        embeddings = get_embeddings_batch(texts_to_embed)
        
        # 3. Prepare PointStructs for Upsert
        points_to_upload = []
        for point, embedding in zip(batch_points, embeddings):
            if embedding is None:
                print(f"    ⚠️ Skipping point {point.id} due to embedding error.")
                continue
                
            # Use NAMED vectors for hybrid search schema (V10.11)
            points_to_upload.append(
                PointStruct(
                    id=point.id,
                    vector={"dense": embedding},
                    payload=point.payload
                )
            )
        
        # 4. Upsert this chunk
        if points_to_upload:
            print(f"    🚀 Upserting {len(points_to_upload)} points with new vectors...")
            client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=points_to_upload
            )
            total_processed += len(points_to_upload)
            
        time.sleep(1) # Small delay to be nice to APIs

    print(f"\n✅ Successfully added embeddings to {total_processed} new points!")
    print(f"📊 Total points in collection: {points_with_embeddings + total_processed}")
    print("Qdrant will now automatically build the HNSW index in the background.")

if __name__ == "__main__":
    generate_and_upsert_vectors()
