"""
The Gardener: Memory Decay & Consolidation Service
"""
import os
import time
import asyncio
import math
from datetime import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI

# Load environment variables
load_dotenv()

# --- CONFIG (from environment) ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional for local
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "decay_memory_mcp"

DECAY_RATE = 0.995
DRY_RUN = True  # Set to False to activate

# --- SETUP ---
if not OPENAI_API_KEY:
    exit("Error: OPENAI_API_KEY not set in environment")

if QDRANT_API_KEY:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
openai_client = OpenAI(api_key=OPENAI_API_KEY)

async def main():
    print(f"🌱 GARDENER START (Dry Run: {DRY_RUN})")
    
    # 1. DECAY
    print("\n--- 📉 DECAY ---")
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION_NAME, limit=50, with_payload=True, offset=next_offset,
            scroll_filter=models.Filter(must_not=[models.FieldCondition(key="archived", match=models.MatchValue(value=True))])
        )
        for point in batch:
            score = point.payload.get('score', 1.0)
            last_ts = point.payload.get('last_accessed', time.time())
            
            days = (time.time() - last_ts) / 86400
            if days < 1 or score <= 0.15: continue
            
            new_score = max(0.1, score * (math.pow(DECAY_RATE, days)))
            if abs(score - new_score) > 0.01:
                if not DRY_RUN:
                    client.set_payload(collection_name=COLLECTION_NAME, payload={"score": new_score}, points=[point.id])
                    print(f"   Decayed: {score:.3f} -> {new_score:.3f}")
                else:
                    print(f"   [DRY] Would decay: {score:.3f} -> {new_score:.3f}")
        if next_offset is None: break

    # 2. DREAM (Consolidate)
    print("\n--- 🕸️ DREAM ---")
    filter_cond = models.Filter(must=[
        models.FieldCondition(key="type", match=models.MatchValue(value="dialog")),
        models.FieldCondition(key="score", range=models.Range(lt=0.4)),
        models.FieldCondition(key="archived", match=models.MatchValue(value=False) if not DRY_RUN else None)
    ])
    
    points, _ = client.scroll(collection_name=COLLECTION_NAME, scroll_filter=filter_cond, limit=50, with_payload=True)
    if len(points) > 3:
        text = "\n".join([f"- {p.payload.get('content')}" for p in points])
        if DRY_RUN:
            print(f"   [DRY] Would summarize {len(points)} memories.")
        else:
            print("   🧠 Dreaming...")
            summary = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Summarize these into one insight:\n{text}"}]
            ).choices[0].message.content
            
            emb = openai_client.embeddings.create(model="text-embedding-3-small", input=summary).data[0].embedding
            import uuid
            client.upsert(collection_name=COLLECTION_NAME, points=[
                models.PointStruct(
                    id=str(uuid.uuid4()), vector={"dense": emb},
                    payload={"content": summary, "type": "summary", "score": 1.5, "timestamp": time.time(), "archived": False}
                )
            ])
    print("Gardener Finished.")

if __name__ == "__main__":
    asyncio.run(main())
