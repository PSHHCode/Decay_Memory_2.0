import time
import json
import asyncio
import os
import re
import logging
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path
from collections import defaultdict

# --- LOGGING SETUP ---
logger = logging.getLogger("FlightRecorder")

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).parent.absolute()
FLIGHT_DIR = SCRIPT_DIR / "flight_recorders"

# --- CONFIGURATION (from main config, for now hardcoded or passed in) ---
# These will ideally be loaded from a central config service later
FLIGHT_RECORDER_RETENTION_HOURS = 48
FLIGHT_RECORDER_MAX_TURNS = 100
MAX_TURN_CONTENT_LENGTH = 500 # Used by format_recent_turns in memory_service

# =============================================================================
# CONCURRENCY LOCKS
# =============================================================================

_file_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

def get_file_lock(filename: str) -> asyncio.Lock:
    """Get or create a lock for a specific file."""
    return _file_locks[filename]


# =============================================================================
# FILENAME SANITIZATION (Security)
# =============================================================================

def get_safe_filename(proj: Optional[str], user_id: Optional[str] = None) -> Path:
    """Sanitize project and user ID to create a safe file path."""
    safe_user = re.sub(r'[^\w\s-]', '_', user_id or "global_user")
    user_dir = Path(safe_user)

    name = proj or "global"
    name = re.sub(r'[^\w\s-]', '_', name)
    name = name.strip().replace(' ', '_')
    
    if '..' in name or name.startswith('/') or name.startswith('\\'):
        logger.warning(f"Detected potentially malicious project name: {proj}")
        return user_dir / "flight_recorder_sanitized_input.jsonl"
    return user_dir / f"flight_recorder_{name}.jsonl"


def get_rec_path(proj: Optional[str], user_id: str) -> Path:
    """Get flight recorder path with sanitized, user-specific filename."""
    user_proj_path = get_safe_filename(proj, user_id)
    return FLIGHT_DIR / user_proj_path


# =============================================================================
# FILE I/O OPERATIONS (Async-first with proper locking)
# =============================================================================

def _append_to_file(path: Path, content: str) -> None:
    """Helper to append content to file (sync)."""
    path.parent.mkdir(parents=True, exist_ok=True) # Ensure user directory exists
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
    path.parent.mkdir(parents=True, exist_ok=True) # Ensure user directory exists
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


async def log_turn_async(user_id: str, proj: str, user: str, ai: str) -> None:
    """Log a conversation turn with file locking for a specific user."""
    path = get_rec_path(proj, user_id)
    filename = str(path) 
    async with get_file_lock(filename):
        try:
            loop = asyncio.get_running_loop()
            line = json.dumps({"ts": time.time(), "turn": {"user": user, "ai": ai}}) + "\n"
            await loop.run_in_executor(None, lambda: _append_to_file(path, line))
        except Exception as e:
            logger.error(f"Error logging turn: {e}")


async def update_last_turn_response_async(user_id: str, proj: str, ai_response: str) -> bool:
    """Update the most recent flight recorder entry with file locking for a user."""
    path = get_rec_path(proj, user_id)
    filename = str(path)
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


def read_rec(proj: str, user_id: str) -> List[Dict[str, Any]]:
    """Read flight recorder for a specific user with malformed line handling."""
    p = get_rec_path(proj, user_id)
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


async def cleanup_old_turns_async(user_id: str, proj: str) -> None:
    """Prune old logs for a specific user with file locking."""
    path = get_rec_path(proj, user_id)
    filename = str(path)
    async with get_file_lock(filename):
        try:
            if not path.exists():
                return
            
            loop = asyncio.get_running_loop()
            
            def _prune() -> None:
                turns = read_rec(proj, user_id)
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
                    logger.info(f"Pruned {len(turns) - len(valid_turns)} turns from '{proj or 'global'}' for user '{user_id}'")
            
            await loop.run_in_executor(None, _prune)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def cleanup_old_turns_sync(proj: str, user_id: str) -> None:
    """Sync version for startup cleanup only (no async context available)."""
    p = get_rec_path(proj, user_id)
    if not p.exists():
        return
    try:
        turns = read_rec(proj, user_id)
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
            logger.info(f"Pruned {len(turns) - len(valid_turns)} turns from '{proj or 'global'}' for user '{user_id}'")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Helper to format recent turns (still used by memory_service.get_handover)
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
        fmt.append(f"[ {time_str} ]\nU: {user}\nA: {ai}")
    return "\n\n".join(fmt)
