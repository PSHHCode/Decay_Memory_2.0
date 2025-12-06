"""
Step 2: Generate Embeddings for Imported Memories (CLOUD/REMOTE VERSION)
VERSION: 2025-12-04

Usage:
  Set environment variables or edit the config section below:
  - QDRANT_URL: Your Qdrant endpoint (cloud or droplet)
  - QDRANT_API_KEY: Optional API key for Qdrant Cloud
  - OPENAI_API_KEY: Required for generating embeddings
  
  python step_2_generate_and_upsert_OPTIMIZED_02.py
"""

import os
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from openai import OpenAI

load_dotenv()

# --- CONFIGURATION ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")  # Optional for local Qdrant
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "decay_memory_mcp"
MODEL_NAME = "text-embedding-3-small"
CONTENT_FIELD = "content"
BATCH_SIZE = 64 

# --- VALIDATION ---
if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not set in environment")
    print("Set it with: export OPENAI_API_KEY='your-key' (Linux/Mac)")
    print("         or: $env:OPENAI_API_KEY='your-key' (PowerShell)")
    exit(1)

# --- INIT ---
print("="*60)
print("EMBEDDING GENERATOR (CLOUD/REMOTE)")
print(f"Target: {QDRANT_URL}")
print("="*60)

try:
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(url=QDRANT_URL)
        
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✓ Connected to Qdrant")
    print("✓ OpenAI client initialized")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    exit(1)

def get_embeddings_batch(texts):
    try:
        response = openai_client.embeddings.create(model=MODEL_NAME, input=texts)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"    ✗ Embedding Error: {e}")
        return [None] * len(texts)

def has_embedding(vector):
    if vector is None: return False
    if isinstance(vector, dict):
        dense = vector.get('dense', [])
        return dense and not all(v == 0.0 for v in dense)
    return not all(v == 0.0 for v in vector)

def generate_and_upsert_vectors():
    print(f"\n## Scanning '{COLLECTION_NAME}' for missing vectors...")
    points_to_process = []
    points_with_embeddings = 0
    next_offset = None
    
    while True:
        try:
            scroll_result, next_offset = client.scroll(
                collection_name=COLLECTION_NAME, limit=200, 
                with_payload=True, with_vectors=True, offset=next_offset
            )
        except Exception as e:
            print(f"❌ Scroll Error: {e}")
            break
        
        for point in scroll_result:
            if has_embedding(point.vector): 
                points_with_embeddings += 1
            else: 
                points_to_process.append(point)
        
        if next_offset is None: break

    print(f"   - Already have embeddings: {points_with_embeddings}")
    print(f"   - Need embeddings: {len(points_to_process)}")
    
    if not points_to_process:
        print("\n🎉 All points already have embeddings! Nothing to do.")
        return

    total_processed = 0
    total_batches = (len(points_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(points_to_process), BATCH_SIZE):
        batch = points_to_process[i:i + BATCH_SIZE]
        texts = [p.payload[CONTENT_FIELD] for p in batch if p.payload and CONTENT_FIELD in p.payload]
        
        if not texts: continue
        
        batch_num = i // BATCH_SIZE + 1
        print(f"\n--- Batch {batch_num}/{total_batches} ({len(texts)} points) ---")
        print("    🧠 Generating embeddings...")
        embeddings = get_embeddings_batch(texts)
        
        points_to_upload = []
        for point, emb in zip(batch, embeddings):
            if emb:
                points_to_upload.append(PointStruct(
                    id=point.id, vector={"dense": emb}, payload=point.payload
                ))
        
        if points_to_upload:
            client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload)
            total_processed += len(points_to_upload)
            print(f"    🚀 Upserted {len(points_to_upload)} points")
        
        time.sleep(0.5)  # Rate limiting

    print(f"\n{'='*60}")
    print(f"COMPLETE: Added embeddings to {total_processed} points")
    print(f"Total points in collection: {points_with_embeddings + total_processed}")
    print(f"{'='*60}")

if __name__ == "__main__":
    generate_and_upsert_vectors()
