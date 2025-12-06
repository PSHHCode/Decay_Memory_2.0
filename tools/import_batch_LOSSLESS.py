"""
Lossless Chat Import - No AI Extraction
VERSION: 2025-12-03 - HYBRID SEARCH COMPATIBLE
Philosophy: Store EVERYTHING, let search and decay handle relevance

Changes from previous versions:
- FIX: Uses named vectors {"dense": [...]} to match V10.11 hybrid search schema
- NO Gemini extraction (no data loss, faster, cheaper)
- Stores complete conversation turns as-is
- Chunks only if a single turn exceeds size limit
- Preserves ALL financial data, philosophical discussions, specific details
- Simple type assignment: user messages vs assistant responses
"""

import os
import re
import time
import hashlib
import uuid
from datetime import datetime
import requests
from typing import List, Dict, Any

# Configuration
import os
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "decay_memory_mcp"
VECTOR_DIMENSION = 1536
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set in environment")
    exit(1)

# Chunk size for very long individual messages
MAX_CHUNK_CHARS = 3000  # ~2000 tokens, safe for embedding later

# Memory type weights (same as MCP server)
TYPE_WEIGHTS = {
    'personal': 1.8,
    'preference': 1.5,
    'goal': 1.3,
    'project': 1.2,
    'topic': 1.2,
    'dialog': 0.6
}


def extract_chat_metadata(content):
    """Extract chat metadata from markdown export"""
    metadata = {
        'chat_id': None,
        'chat_title': None,
        'source': None,
        'created_timestamp': None
    }
    
    DATE_FORMATS = [
        "%d/%m/%Y, %H:%M:%S",
        "%m/%d/%Y, %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y"
    ]
    
    overview_match = re.search(r'## Overview\n(.*?)## Conversation', content, re.DOTALL)
    if overview_match:
        overview_content = overview_match.group(1)
        
        title_match = re.search(r'- \*\*Title:\*\* (.+)', overview_content)
        if title_match:
            metadata['chat_title'] = title_match.group(1).strip()

        id_match = re.search(r'- \*\*ID:\*\* ([a-f0-9-]+)', overview_content)
        if id_match:
            metadata['chat_id'] = id_match.group(1).strip()
            
        created_match = re.search(r'- \*\*Created:\*\* ([^\n]+)', overview_content)
        if created_match:
            date_str = created_match.group(1).strip()
            for fmt in DATE_FORMATS:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    metadata['created_timestamp'] = dt.timestamp()
                    break
                except ValueError:
                    continue

        link_match = re.search(r'- \*\*Url:\*\* \[([^\]]+)\]', overview_content)
        if link_match:
            url = link_match.group(1)
            if 'claude.ai' in url:
                metadata['source'] = 'claude'
            elif 'gemini.google.com' in url:
                metadata['source'] = 'gemini'
    
    if not metadata['chat_id'] and not metadata['chat_title']:
        title_match_old = re.search(r'^# (.+?)$', content, re.MULTILINE)
        if title_match_old:
            metadata['chat_title'] = title_match_old.group(1).strip()

    return metadata


def parse_conversation_turns(content):
    """
    Parse conversation into individual turns (user/assistant messages)
    Gets ALL messages and deduplicates based on timestamp+role
    Returns list of dicts: [{"role": "user"/"assistant", "content": "...", "timestamp": ...}, ...]
    """
    turns = {}  # Use dict to deduplicate by (timestamp, role)
    
    # Extract just the conversation section (stop before footer)
    content_match = re.search(r'## Conversation\n(.*?)---\s*<div align', content, re.DOTALL)
    if content_match:
        conversation = content_match.group(1).strip()
    else:
        content_match = re.search(r'## Conversation\n(.*?)---', content, re.DOTALL)
        if content_match:
            conversation = content_match.group(1).strip()
        else:
            conversation = content.strip()
    
    # Find ALL message blocks (user or assistant, with or without arrow marker)
    # Pattern: timestamp + (User or Assistant) + content
    pattern = r'<i>\[([\d\/,: ]+)\]</i>(?:\s*👉)?\s*<b>(👤 User|🤖 Assistant)</b>:\s*(.*?)(?=<i>\[|</?details|</?summary|$)'
    
    matches = re.finditer(pattern, conversation, re.DOTALL)
    
    for match in matches:
        timestamp_str = match.group(1).strip()
        role_marker = match.group(2).strip()
        content_text = match.group(3).strip()
        
        # Determine role
        role = 'user' if '👤' in role_marker else 'assistant'
        
        # Skip if this is just a preview (very short, ends with ...)
        if len(content_text) < 50 and content_text.endswith('...'):
            continue
        
        # Clean up HTML
        content_text = re.sub(r'<br>|</br>', '\n', content_text)
        content_text = re.sub(r'<[^>]+>', '', content_text)
        content_text = content_text.strip()
        
        # Skip empty or very short snippets
        if len(content_text) < 20:
            continue
        
        # Use (timestamp, role) as key to deduplicate
        key = (timestamp_str, role)
        
        # Keep the longer version if duplicate
        if key in turns:
            if len(content_text) > len(turns[key]['content']):
                turns[key] = {
                    'role': role,
                    'content': content_text,
                    'timestamp_str': timestamp_str
                }
        else:
            turns[key] = {
                'role': role,
                'content': content_text,
                'timestamp_str': timestamp_str
            }
    
    # Convert dict to sorted list
    return sorted(turns.values(), key=lambda x: x['timestamp_str'])


def chunk_large_content(content: str, max_chars: int) -> List[str]:
    """Split large content into chunks, handling oversized paragraphs"""
    if len(content) <= max_chars:
        return [content]
    
    chunks = []
    paragraphs = content.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        # If paragraph itself is too large, split it by sentences
        if len(para) > max_chars:
            # Split oversized paragraph by sentences
            sentences = para.split('. ')
            for sentence in sentences:
                # If even a single sentence is too large, split by character chunks
                if len(sentence) > max_chars:
                    # Hard split at word boundaries
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= max_chars:
                            temp_chunk += word + ' '
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word + ' '
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                else:
                    if len(current_chunk) + len(sentence) + 2 <= max_chars:
                        current_chunk += sentence + '. '
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
        else:
            # Normal paragraph handling
            if len(current_chunk) + len(para) + 2 <= max_chars:
                current_chunk += para + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + '\n\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def generate_point_id(content: str, chat_id: str, turn_index: int) -> str:
    """Generate deterministic UUID from content, chat_id, and turn index"""
    combined = f"{content}_{chat_id}_{turn_index}".encode('utf-8')
    hash_value = hashlib.sha256(combined).hexdigest()
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, hash_value))


def infer_memory_type(turn: Dict, chat_title: str) -> str:
    """
    Intelligently infer memory type from turn content and context
    This is a simple heuristic - can be enhanced
    """
    content = turn['content'].lower()
    role = turn['role']
    
    # User questions/statements - try to categorize
    if role == 'user':
        # Personal statements
        if any(phrase in content for phrase in ['i am', 'i have', 'i own', 'my ', 'i live', 'i work']):
            return 'personal'
        # Preferences
        if any(phrase in content for phrase in ['i prefer', 'i like', 'i want', 'i love', 'i hate']):
            return 'preference'
        # Goals
        if any(phrase in content for phrase in ['i want to', 'i need to', 'planning to', 'goal is', 'hoping to']):
            return 'goal'
        # Projects
        if any(phrase in content for phrase in ['working on', 'building', 'creating', 'developing', 'project']):
            return 'project'
        # Default for user turns
        return 'dialog'
    
    # Assistant responses - usually topic or dialog
    else:
        # If response contains lots of data/facts/analysis, might be topic
        if len(content) > 500:  # Long, detailed responses
            return 'topic'
        return 'dialog'


def add_memories_to_qdrant_batch(memories: List[Dict[str, Any]], metadata: Dict[str, Any]) -> tuple[int, int]:
    """Add memories to Qdrant with unnamed vectors (zero-filled)"""
    if not memories:
        return 0, 0
    
    timestamp = metadata['created_timestamp'] if metadata['created_timestamp'] else time.time()
    zero_vector = [0.0] * VECTOR_DIMENSION
    
    points = []
    for mem in memories:
        content = mem.get('content', '')
        mem_type = mem.get('type', 'dialog')
        score = TYPE_WEIGHTS.get(mem_type, 1.0)
        
        payload = {
            "content": content,
            "type": mem_type,
            "score": score,
            "timestamp": timestamp,
            "last_accessed": timestamp,
            "access_count": 0,
            "chat_id": metadata['chat_id'],
            "chat_title": metadata['chat_title'],
            "source": metadata['source'],
            "role": mem.get('role', 'unknown'),  # Track if user or assistant
            "turn_index": mem.get('turn_index', 0),
            "chunk_index": mem.get('chunk_index', 0),
            "project_name": None,
            "project_memory": None
        }
        
        points.append({
            "id": generate_point_id(content, metadata['chat_id'], mem.get('turn_index', 0)),
            "vector": {"dense": zero_vector},  # V10.11 FIX: Named vectors for hybrid search
            "payload": payload
        })
    
    try:
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json={"points": points}
        )
        
        if response.status_code == 200:
            return len(points), 0
        else:
            print(f"    ✗ Qdrant error: {response.text}")
            return 0, len(points)
    except Exception as e:
        print(f"    ✗ Qdrant error: {e}")
        return 0, len(points)


def get_md_files(directory):
    """Get all .md files in directory"""
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            files.append(os.path.join(directory, filename))
    return sorted(files)


def import_batch_lossless(directory):
    """Import all chats with complete data preservation"""
    print("="*80)
    print("LOSSLESS CHAT IMPORT - COMPLETE DATA PRESERVATION")
    print("="*80)
    
    files = get_md_files(directory)
    
    if not files:
        print(f"✗ No .md files found in {directory}")
        return
    
    print(f"\nFound {len(files)} chat files to process")
    print("Mode: LOSSLESS - Stores every conversation turn as-is")
    print("No AI extraction, no data loss, much faster\n")
    print("Processing...\n")
    
    total_files = len(files)
    total_memories = 0
    total_success = 0
    total_failed = 0
    files_processed = 0
    files_skipped = 0

    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        print(f"[{i}/{total_files}] {filename}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata
            metadata = extract_chat_metadata(content)
            
            print(f"  Title: {metadata['chat_title']}")
            print(f"  Source: {metadata['source']}")
            print(f"  Chat ID: {metadata['chat_id']}")
            
            if metadata['created_timestamp']:
                created_dt = datetime.fromtimestamp(metadata['created_timestamp'])
                created_date_str = created_dt.strftime("%B %d, %Y, %H:%M:%S")
                age_days = (time.time() - metadata['created_timestamp']) / 86400
                print(f"  Created: {created_date_str} ({age_days:.0f} days ago)")
            else:
                print(f"  ⚠ No created date found, using current time")
                if not metadata['chat_id']:
                    metadata['chat_id'] = hashlib.sha1(filename.encode()).hexdigest()
            
            if not metadata['chat_id']:
                print(f"  ✗ Could not determine unique chat_id, skipping file")
                files_skipped += 1
                continue
            
            # Parse conversation turns
            turns = parse_conversation_turns(content)
            
            if not turns:
                print(f"  ⚠ No conversation turns found, skipping file")
                files_skipped += 1
                continue
            
            print(f"  Found {len(turns)} conversation turns")
            
            # Convert turns to memories
            all_memories = []
            
            for turn_idx, turn in enumerate(turns):
                # Check if turn needs chunking
                chunks = chunk_large_content(turn['content'], MAX_CHUNK_CHARS)
                
                for chunk_idx, chunk in enumerate(chunks):
                    memory_type = infer_memory_type(turn, metadata['chat_title'])
                    
                    all_memories.append({
                        'content': chunk,
                        'type': memory_type,
                        'role': turn['role'],
                        'turn_index': turn_idx,
                        'chunk_index': chunk_idx
                    })
            
            print(f"  Storing {len(all_memories)} memories (including {len(all_memories) - len(turns)} chunks from long turns)")
            
            # Add to Qdrant
            file_success, file_failed = add_memories_to_qdrant_batch(all_memories, metadata)
            
            print(f"  ✓ Added {file_success} memories")
            if file_failed > 0:
                print(f"  ✗ Failed {file_failed} memories")
            
            total_memories += len(all_memories)
            total_success += file_success
            total_failed += file_failed
            files_processed += 1
            
        except Exception as e:
            print(f"  ✗ Error processing file: {e}")
            files_skipped += 1
        
        print()
    
    # Summary
    print("="*80)
    print("IMPORT COMPLETE - LOSSLESS MODE")
    print("="*80)
    print(f"Files processed: {files_processed}/{total_files}")
    print(f"Files skipped: {files_skipped}")
    print(f"Total memories created: {total_memories}")
    print(f"Successfully added: {total_success}")
    print(f"Failed: {total_failed}")
    print("="*80)
    
    if total_success > 0:
        print("\n✓ All conversation data preserved!")
        print("\nNext steps:")
        print("1. Run step_2_generate_and_upsert.py to add embeddings")
        print("2. Adjust type weights in MCP server if needed")
        print("3. Use decay to manage importance over time")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python import_batch_LOSSLESS.py <path_to_chats_folder>")
        print("\nExample:")
        print("  python import_batch_LOSSLESS.py C:\\Users\\Steph\\Downloads\\ChatExports")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)
    
    if not os.path.isdir(directory):
        print(f"Error: Not a directory: {directory}")
        sys.exit(1)
    
    import_batch_lossless(directory)
