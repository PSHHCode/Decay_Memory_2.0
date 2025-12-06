"""
Lossless Chat Import - No AI Extraction (CLOUD/REMOTE VERSION)
VERSION: 2025-12-04 - HYBRID SEARCH COMPATIBLE

Usage:
  Set environment variables or edit the config section below:
  - QDRANT_URL: Your Qdrant endpoint (cloud or droplet)
  - QDRANT_API_KEY: Optional API key for Qdrant Cloud
  
  python import_batch_LOSSLESS_02.py /path/to/chat/exports
"""

import os
import re
import time
import hashlib
import uuid
from datetime import datetime
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURATION - Uses env vars with fallbacks
# ==========================================
# Override with environment variables or edit defaults below
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")  # Optional for local Qdrant

COLLECTION_NAME = "decay_memory_mcp"
VECTOR_DIMENSION = 1536

# Chunk size for very long individual messages
MAX_CHUNK_CHARS = 3000  

# Memory type weights
TYPE_WEIGHTS = {
    'personal': 1.8, 'preference': 1.5, 'goal': 1.3,
    'project': 1.2, 'topic': 1.2, 'dialog': 0.6
}

def extract_chat_metadata(content):
    metadata = {'chat_id': None, 'chat_title': None, 'source': None, 'created_timestamp': None}
    DATE_FORMATS = ["%d/%m/%Y, %H:%M:%S", "%m/%d/%Y, %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"]
    
    overview_match = re.search(r'## Overview\n(.*?)## Conversation', content, re.DOTALL)
    if overview_match:
        overview_content = overview_match.group(1)
        title_match = re.search(r'- \*\*Title:\*\* (.+)', overview_content)
        if title_match: metadata['chat_title'] = title_match.group(1).strip()

        id_match = re.search(r'- \*\*ID:\*\* ([a-f0-9-]+)', overview_content)
        if id_match: metadata['chat_id'] = id_match.group(1).strip()
            
        created_match = re.search(r'- \*\*Created:\*\* ([^\n]+)', overview_content)
        if created_match:
            date_str = created_match.group(1).strip()
            for fmt in DATE_FORMATS:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    metadata['created_timestamp'] = dt.timestamp()
                    break
                except ValueError: continue

        link_match = re.search(r'- \*\*Url:\*\* \[([^\]]+)\]', overview_content)
        if link_match:
            url = link_match.group(1)
            if 'claude.ai' in url: metadata['source'] = 'claude'
            elif 'gemini.google.com' in url: metadata['source'] = 'gemini'
    
    if not metadata['chat_id'] and not metadata['chat_title']:
        title_match_old = re.search(r'^# (.+?)$', content, re.MULTILINE)
        if title_match_old: metadata['chat_title'] = title_match_old.group(1).strip()
    return metadata

def parse_conversation_turns(content):
    turns = {}
    content_match = re.search(r'## Conversation\n(.*?)---\s*<div align', content, re.DOTALL)
    if not content_match:
        content_match = re.search(r'## Conversation\n(.*?)---', content, re.DOTALL)
    conversation = content_match.group(1).strip() if content_match else content.strip()
    
    pattern = r'<i>\[([\d\/,: ]+)\]</i>(?:\s*👉)?\s*<b>(👤 User|🤖 Assistant)</b>:\s*(.*?)(?=<i>\[|</?details|</?summary|$)'
    matches = re.finditer(pattern, conversation, re.DOTALL)
    
    for match in matches:
        timestamp_str = match.group(1).strip()
        role = 'user' if '👤' in match.group(2) else 'assistant'
        content_text = match.group(3).strip()
        
        if len(content_text) < 50 and content_text.endswith('...'): continue
        content_text = re.sub(r'<br>|</br>', '\n', content_text)
        content_text = re.sub(r'<[^>]+>', '', content_text).strip()
        
        if len(content_text) < 20: continue
        
        key = (timestamp_str, role)
        if key in turns:
            if len(content_text) > len(turns[key]['content']):
                turns[key] = {'role': role, 'content': content_text, 'timestamp_str': timestamp_str}
        else:
            turns[key] = {'role': role, 'content': content_text, 'timestamp_str': timestamp_str}
    
    return sorted(turns.values(), key=lambda x: x['timestamp_str'])

def chunk_large_content(content: str, max_chars: int) -> List[str]:
    if len(content) <= max_chars: return [content]
    chunks = []
    paragraphs = content.split('\n\n')
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += para + '\n\n'
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = para + '\n\n'
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def generate_point_id(content: str, chat_id: str, turn_index: int) -> str:
    combined = f"{content}_{chat_id}_{turn_index}".encode('utf-8')
    hash_value = hashlib.sha256(combined).hexdigest()
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, hash_value))

def infer_memory_type(turn: Dict, chat_title: str) -> str:
    content = turn['content'].lower()
    if turn['role'] == 'user':
        if any(x in content for x in ['i am', 'i have', 'i own', 'my ', 'i live']): return 'personal'
        if any(x in content for x in ['prefer', 'like', 'want', 'love', 'hate']): return 'preference'
        if any(x in content for x in ['want to', 'need to', 'planning', 'goal']): return 'goal'
        if any(x in content for x in ['working on', 'building', 'project']): return 'project'
        return 'dialog'
    else:
        return 'topic' if len(content) > 500 else 'dialog'

def add_memories_to_qdrant_batch(memories: List[Dict[str, Any]], metadata: Dict[str, Any]) -> tuple[int, int]:
    if not memories: return 0, 0
    timestamp = metadata['created_timestamp'] if metadata['created_timestamp'] else time.time()
    points = []
    
    for mem in memories:
        payload = {
            "content": mem['content'], "type": mem['type'], "score": TYPE_WEIGHTS.get(mem['type'], 1.0),
            "timestamp": timestamp, "last_accessed": timestamp, "access_count": 0,
            "chat_id": metadata['chat_id'], "chat_title": metadata['chat_title'], "source": metadata['source'],
            "role": mem['role'], "turn_index": mem['turn_index'], "chunk_index": mem['chunk_index'],
            "project_name": None, "project_memory": None
        }
        points.append({
            "id": generate_point_id(mem['content'], metadata['chat_id'], mem['turn_index']),
            "vector": {"dense": [0.0] * VECTOR_DIMENSION},
            "payload": payload
        })
    
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
        
    try:
        response = requests.put(f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points", json={"points": points}, headers=headers)
        return (len(points), 0) if response.status_code == 200 else (0, len(points))
    except Exception as e:
        print(f"    ✗ Connection error: {e}")
        return 0, len(points)

def get_md_files(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.md')])

def import_batch_lossless(directory):
    print(f"="*60)
    print(f"LOSSLESS IMPORT (CLOUD/REMOTE)")
    print(f"Target: {QDRANT_URL}")
    print(f"="*60)

    files = get_md_files(directory)
    if not files:
        print(f"No .md files found in {directory}")
        return
    
    print(f"Found {len(files)} files to process\n")
    
    total_memories = 0
    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        print(f"[{i}/{len(files)}] {filename}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
            metadata = extract_chat_metadata(content)
            if not metadata['chat_id']: metadata['chat_id'] = hashlib.sha1(filename.encode()).hexdigest()
            
            turns = parse_conversation_turns(content)
            all_memories = []
            for t_idx, turn in enumerate(turns):
                chunks = chunk_large_content(turn['content'], MAX_CHUNK_CHARS)
                for c_idx, chunk in enumerate(chunks):
                    all_memories.append({
                        'content': chunk, 'type': infer_memory_type(turn, metadata['chat_title']),
                        'role': turn['role'], 'turn_index': t_idx, 'chunk_index': c_idx
                    })
            
            success, _ = add_memories_to_qdrant_batch(all_memories, metadata)
            print(f"  ✓ Added {success} memories")
            total_memories += success
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_memories} total memories imported")
    print(f"Next: Run step_2_generate_and_upsert_OPTIMIZED_02.py to add embeddings")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python import_batch_LOSSLESS_02.py <path_to_chats>")
        print("\nEnvironment variables:")
        print("  QDRANT_URL - Qdrant endpoint (default: http://localhost:6333)")
        print("  QDRANT_API_KEY - Optional API key for Qdrant Cloud")
        sys.exit(1)
    import_batch_lossless(sys.argv[1])
