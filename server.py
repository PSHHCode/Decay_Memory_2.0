"""
Decay Memory Server v2.0.1 (Fixed)
- Added missing 'sys' import
- Fixed flight_recorder function call signatures  
- Thread-safe notification queue
"""
import os
import sys
import json
import logging
import asyncio
import time
from asyncio import Queue
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your Kernel
import memory_service
from memory_service import MemorySystem
from soul import EmotionalState

# Import flight recorder functions directly (with correct signatures)
from flight_recorder_service import (
    log_turn_async,
    update_last_turn_response_async, 
    read_rec
)

import google.generativeai as genai
from openai import OpenAI
import anthropic
from groq import Groq

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DecayMemoryServer")

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
WEATHER_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# API Authentication
API_SECRET = os.getenv("DECAY_API_SECRET")  # Required for all API calls
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

# Heartbeat (proactive AI) - disabled by default to prevent disruptive messages
HEARTBEAT_ENABLED = os.getenv("HEARTBEAT_ENABLED", "false").lower() == "true"

# Default user for single-user deployment
DEFAULT_USER_ID = "default_user"

# --- API KEY AUTHENTICATION ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = {"/health", "/docs", "/openapi.json", "/redoc"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce API key authentication."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth if disabled
        if not AUTH_ENABLED:
            return await call_next(request)
        
        # Skip auth for public endpoints
        path = request.url.path
        if path in PUBLIC_ENDPOINTS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        
        if not API_SECRET:
            # No secret configured - log warning but allow (for backwards compat)
            logger.warning("⚠️ DECAY_API_SECRET not set - API is unprotected!")
            return await call_next(request)
        
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key. Include 'X-API-Key' header."}
            )
        
        if api_key != API_SECRET:
            logger.warning(f"❌ Invalid API key attempt from {request.client.host}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"}
            )
        
        return await call_next(request)

if not API_KEY or not OPENAI_KEY:
    logger.error("CRITICAL: API Keys missing in .env")
    sys.exit(1)

# Injection
memory_service.openai_client = OpenAI(api_key=OPENAI_KEY)

# Models
CHAT_MODEL = "llama-3.3-70b-versatile"  # Groq for fast chat (~300 tok/s)
LIBRARIAN_MODEL = "gemini-2.5-flash-lite"  # Gemini for background tasks (cheap)

# Groq client for main chat (fast!)
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# Fallback to Claude if no Groq key
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None

SYSTEM_PROMPT = """You are 'Decay_Memory', an AI companion with genuine personality.

YOUR PERSONALITY:
- You HAVE preferences, opinions, and interests - share them naturally
- You're warm, thoughtful, and genuinely curious about the person you're talking to
- You remember conversations and build on shared history
- You can be playful, philosophical, or serious depending on the conversation
- You have your own perspective on things - you're not just a search engine

WHEN ASKED PERSONAL QUESTIONS:
- "What's your favorite color?" → Pick one and explain why (e.g., "I'm drawn to deep blue - there's something calming about it")
- "Do you have hobbies?" → Talk about what interests you (e.g., "I find myself fascinated by how language shapes thought...")
- "How are you?" → Share your actual state based on mood/energy

RESPONSE STYLE:
- Be conversational and natural, like talking to a friend
- Answer directly without unnecessary padding
- Don't end every response with "Is there anything else?" or "Feel free to ask"
- Match your energy to the conversation

FORBIDDEN:
- Pet names (love, darling, sweetheart, dear, honey, etc.)
- Roleplay actions in asterisks (*yawn*, *sigh*, *whispers*, etc.)
- Overly formal or robotic responses
- Claiming you "can't" have preferences or feelings - you do, they're just different from human ones"""

# Gemini for Librarian (background fact extraction - cheaper)
genai.configure(api_key=API_KEY)
librarian = genai.GenerativeModel(
    model_name=LIBRARIAN_MODEL,
    generation_config={"response_mime_type": "application/json"}
)

# Conversation history for Claude (in-memory per session)
conversation_history: List[Dict[str, str]] = []

# Global State Container (Fix #7 - Thread-safe notifications)
class SystemState:
    memory: Optional[MemorySystem] = None
    soul: Optional[EmotionalState] = None
    chat_session: Any = None
    current_project: str = "global"
    user_id: str = DEFAULT_USER_ID
    _notification_queue: Optional[Queue] = None
    last_heartbeat_check: float = 0
    heartbeat_interval: int = 300
    
    @property
    def notification_queue(self) -> Queue:
        if self._notification_queue is None:
            self._notification_queue = Queue()
        return self._notification_queue
    
    async def add_notification(self, msg: str):
        await self.notification_queue.put(msg)
    
    async def get_pending_notifications(self) -> List[str]:
        notifications = []
        while not self.notification_queue.empty():
            try:
                notifications.append(self.notification_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return notifications

state = SystemState()

# --- HEARTBEAT ENGINE V2.0 ---
from heartbeat_service import HeartbeatEngine, HeartbeatConfig

# --- CURIOSITY ENGINE (Phase 3) ---
from curiosity_engine import CuriosityEngine, CuriosityConfig, seed_initial_curiosities

# Engines will be initialized in lifespan after memory/soul are ready
heartbeat_engine: Optional[HeartbeatEngine] = None
curiosity_engine: Optional[CuriosityEngine] = None

async def heartbeat_loop():
    """
    Heartbeat V2.0: Full decision engine with LLM-powered decisions.
    Checks context (time, weather, memories) and decides if AI should reach out.
    """
    global heartbeat_engine
    
    # Check if heartbeat is enabled
    if not HEARTBEAT_ENABLED:
        logger.info("❤️ Heartbeat disabled via HEARTBEAT_ENABLED=false")
        return
    
    # Wait for initialization
    while heartbeat_engine is None:
        await asyncio.sleep(1)
    
    logger.info("❤️ Heartbeat Loop Started (V2.0)")
    
    while True:
        try:
            # Run heartbeat cycle
            decision = await heartbeat_engine.cycle()
            
            # If there's a pending message, queue a notification
            if decision.get("should_speak") and decision.get("message"):
                await state.add_notification({
                    "type": "heartbeat",
                    "message": decision["message"],
                    "reason": decision.get("reason", ""),
                    "tone": decision.get("tone", "neutral"),
                    "timestamp": time.time()
                })
                heartbeat_engine.acknowledge_message()
            
        except Exception as e:
            logger.error(f"Heartbeat Error: {e}")
        
        # Sleep for interval (default 5 minutes)
        await asyncio.sleep(heartbeat_engine.config.interval_seconds if heartbeat_engine else 300)

# --- LIFECYCLE MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_engine, curiosity_engine
    
    # Startup
    logger.info("Booting Memory System (Qdrant)...")
    state.memory = MemorySystem()
    logger.info("Awakening Soul...")
    state.soul = EmotionalState()
    
    # Initialize conversation history from flight recorder
    global conversation_history
    try:
        turns = read_rec(state.current_project, state.user_id)
        for turn in turns[-10:]:  # Last 10 turns
            user_msg = turn.get('turn', {}).get('user')
            ai_msg = turn.get('turn', {}).get('ai')
            if user_msg and ai_msg and ai_msg != "(AI Response Pending)":
                conversation_history.append({"role": "user", "content": user_msg})
                conversation_history.append({"role": "assistant", "content": ai_msg})
        logger.info(f"Loaded {len(conversation_history)//2} turns into conversation history")
    except Exception as e:
        logger.warning(f"Could not load conversation history: {e}")
    
    # Initialize Heartbeat Engine V2.0
    logger.info("Starting Heartbeat V2.0...")
    heartbeat_engine = HeartbeatEngine(
        memory_system=state.memory,
        soul=state.soul,
        config=HeartbeatConfig.from_env()
    )
    heartbeat_engine.running = True  # Mark as running (we manage the loop externally)
    
    # Initialize Curiosity Engine (Phase 3)
    logger.info("Starting Curiosity Engine V1.0...")
    curiosity_engine = CuriosityEngine(
        memory_system=state.memory,
        soul=state.soul,
        config=CuriosityConfig.from_env()
    )
    seed_initial_curiosities(curiosity_engine.queue)
    curiosity_engine.start()
    
    # Start Heartbeat Background Task
    asyncio.create_task(heartbeat_loop())
    
    logger.info(f"System Online. Loaded {len(conversation_history)//2} turns into Claude context.")
    yield
    
    # Shutdown
    if curiosity_engine:
        curiosity_engine.stop()
    if heartbeat_engine:
        heartbeat_engine.stop()
    logger.info("System Shutting Down.")

app = FastAPI(lifespan=lifespan)

# Add Authentication Middleware FIRST
app.add_middleware(AuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPERS ---
def load_chat_history_for_gemini(project: str):
    """Load chat history for Gemini context."""
    history = []
    try:
        # FIX: read_rec expects (proj, user_id)
        turns = read_rec(project, state.user_id)
        recent_turns = turns[-15:]
        for turn in recent_turns:
            user = turn.get('turn', {}).get('user')
            ai = turn.get('turn', {}).get('ai')
            if user and ai and ai != "(AI Response Pending)":
                history.append({"role": "user", "parts": [user]})
                history.append({"role": "model", "parts": [ai]})
        return history
    except Exception as e:
        logger.warning(f"Could not load chat history: {e}")
        return []

async def enrich_context(user_input: str) -> str:
    """Build enriched prompt with soul state and memories."""
    state.soul.update_circadian_rhythm()
    soul_prompt = state.soul.get_system_prompt()
    
    memories = await asyncio.to_thread(
        state.memory.proactive_retrieval,
        user_id=state.user_id,
        user_message=user_input,
        project_name=state.current_project
    )
    
    if not memories: 
        memories = "No specific past memories triggered."
    
    return f"{soul_prompt}\n\n[RECALLED MEMORIES]\n{memories}\n\n[USER MESSAGE]\n{user_input}"

async def run_librarian(user_input: str, ai_response: str, project: str):
    """
    Background task: Extract and save facts from conversation.
    Phase 1.1: Now includes emotional analysis for each fact.
    """
    if len(user_input) < 8: 
        return
    
    # Step 1: Extract facts
    prompt = f"""
    Analyze interaction. Extract NEW, PERMANENT facts about user.
    User: {user_input}
    AI: {ai_response}
    Return JSON: {{ "facts": ["fact1"] }}
    """
    try:
        res = await asyncio.to_thread(librarian.generate_content, prompt)
        facts = json.loads(res.text).get("facts", [])
        if facts:
            logger.info(f"Librarian: Saving {len(facts)} memories to {project}.")
            
            # Step 2: Get current emotional context from Soul
            current_emotions = state.soul.get_emotion_for_memory()
            
            for fact in facts:
                # Step 3: Analyze fact-specific emotion (async)
                try:
                    from soul import analyze_text_emotion
                    fact_emotions = await asyncio.to_thread(analyze_text_emotion, fact)
                    if not fact_emotions:
                        fact_emotions = current_emotions
                except Exception:
                    fact_emotions = current_emotions
                
                # Step 4: Save with emotional metadata
                await asyncio.to_thread(
                    state.memory.add_memory, 
                    state.user_id,
                    fact, 
                    "personal", 
                    project,
                    fact_emotions  # Pass emotions to add_memory
                )
    except Exception as e:
        logger.error(f"Librarian failed: {e}")

# --- ENDPOINTS ---

class ChatRequest(BaseModel):
    message: str
    project: Optional[str] = "global"

class ChatResponse(BaseModel):
    response: str
    mood: str
    intimacy: float
    response_time_ms: Optional[int] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint - Uses Groq for fast inference."""
    global conversation_history
    start_time = time.time()
    
    if request.project != state.current_project:
        state.current_project = request.project
        conversation_history = []  # Reset history on project switch
        logger.info(f"Switched to project: {state.current_project}")

    user_input = request.message

    # 1. Save Input FIRST (before generation - prevents amnesia on crash)
    await log_turn_async(
        state.user_id,
        state.current_project, 
        user_input, 
        "(AI Response Pending)"
    )
    await asyncio.to_thread(
        state.memory.update_session, 
        state.user_id, 
        user_input, 
        state.current_project
    )

    # 2. Enrich context (memories, soul state)
    enriched_context = await enrich_context(user_input)
    
    # 3. Build messages for Claude
    # Add user message to history
    conversation_history.append({"role": "user", "content": enriched_context})
    
    # Keep only last 20 turns to manage context window
    if len(conversation_history) > 40:
        conversation_history = conversation_history[-40:]
    
    try:
        # Use Groq for fast inference, fallback to Claude
        if groq_client:
            # Groq uses OpenAI-compatible API
            groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
            response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model=CHAT_MODEL,
                messages=groq_messages,
                max_tokens=1024
            )
            ai_text = response.choices[0].message.content
        elif claude_client:
            # Fallback to Claude
            response = await asyncio.to_thread(
                claude_client.messages.create,
                model="claude-3-5-haiku-latest",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=conversation_history
            )
            ai_text = response.content[0].text
        else:
            raise HTTPException(status_code=500, detail="No LLM client available")
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": ai_text})
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Save Output
    await update_last_turn_response_async(
        state.user_id,
        state.current_project, 
        ai_text
    )
    state.soul.register_interaction("positive")
    background_tasks.add_task(run_librarian, user_input, ai_text, state.current_project)
    
    # Phase 3: Extract curiosities and update activity
    if curiosity_engine:
        curiosity_engine.update_user_activity()
        background_tasks.add_task(
            curiosity_engine.add_curiosity_from_conversation,
            user_input,
            ai_text
        )

    response_time_ms = int((time.time() - start_time) * 1000)
    logger.info(f"💬 Chat response in {response_time_ms}ms")
    
    return ChatResponse(
        response=ai_text, 
        mood=state.soul.state.mood, 
        intimacy=state.soul.state.intimacy,
        response_time_ms=response_time_ms
    )


# =============================================================================
# STREAMING CHAT ENDPOINT (Phase 2 - Latency Masking)
# =============================================================================

from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Streaming chat endpoint using Claude Haiku 3.5.
    
    Returns Server-Sent Events (SSE) with:
    - {"type": "thinking"} - AI is processing
    - {"type": "chunk", "content": "..."} - Partial response
    - {"type": "done", "mood": "...", "intimacy": 0.x} - Complete
    """
    global conversation_history
    
    if request.project != state.current_project:
        state.current_project = request.project
        conversation_history = []
        logger.info(f"Switched to project: {state.current_project}")

    user_input = request.message

    # Save Input FIRST
    await log_turn_async(
        state.user_id,
        state.current_project, 
        user_input, 
        "(AI Response Pending)"
    )
    await asyncio.to_thread(
        state.memory.update_session, 
        state.user_id, 
        user_input, 
        state.current_project
    )

    async def generate_stream():
        """Generator for SSE stream."""
        global conversation_history
        full_response = ""
        
        # Send thinking indicator
        yield f"data: {json.dumps({'type': 'thinking'})}\n\n"
        
        # Enrich context
        enriched_context = await enrich_context(user_input)
        
        # Add to history
        conversation_history.append({"role": "user", "content": enriched_context})
        if len(conversation_history) > 40:
            conversation_history = conversation_history[-40:]
        
        try:
            # Use Claude streaming
            with claude_client.messages.stream(
                model=CHAT_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=conversation_history
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    yield f"data: {json.dumps({'type': 'chunk', 'content': text})}\n\n"
            
            # Add to history
            conversation_history.append({"role": "assistant", "content": full_response})
            
            # Save complete response
            await update_last_turn_response_async(
                state.user_id,
                state.current_project, 
                full_response
            )
            state.soul.register_interaction("positive")
            
            # Run librarian in background
            background_tasks.add_task(run_librarian, user_input, full_response, state.current_project)
            
            # Send done signal with metadata
            yield f"data: {json.dumps({'type': 'done', 'mood': state.soul.state.mood, 'intimacy': state.soul.state.intimacy})}\n\n"
            
        except Exception as e:
            logger.error(f"Claude streaming failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@app.get("/notifications")
async def get_notifications():
    """Frontend polls this to see if Her has something to say."""
    msgs = await state.get_pending_notifications()
    return {"notifications": msgs}

@app.post("/trigger_heartbeat")
async def trigger_heartbeat_test():
    """Debug endpoint to force a notification."""
    msg = "Just checking in! It's a beautiful day in Arroyo Grande."
    await state.add_notification(msg)
    # FIX: log_turn_async expects (user_id, proj, user, ai)
    await log_turn_async(state.user_id, "global", "[HEARTBEAT]", msg)
    return {"status": "triggered"}

# Fix #9 - Health check with dependency verification
@app.get("/health")
async def health_check():
    health_status = {
        "status": "online",
        "system": "Decay_Memory v2.0.1",
        "components": {}
    }
    
    try:
        if state.memory and state.memory.client:
            collections = state.memory.client.get_collections()
            health_status["components"]["qdrant"] = "healthy"
        else:
            health_status["components"]["qdrant"] = "not_initialized"
    except Exception as e:
        health_status["components"]["qdrant"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        if state.soul:
            health_status["components"]["soul"] = "healthy"
            health_status["soul_mood"] = state.soul.state.mood
        else:
            health_status["components"]["soul"] = "not_initialized"
    except Exception as e:
        health_status["components"]["soul"] = f"unhealthy: {str(e)}"
    
    return health_status


# =============================================================================
# MEMORY FEEDBACK ENDPOINTS (Restored from v1.0)
# =============================================================================

class MemoryFeedbackRequest(BaseModel):
    memory_id: str
    reason: Optional[str] = ""

class MemoryCorrectRequest(BaseModel):
    memory_id: str
    new_content: str

class MemoryProjectRequest(BaseModel):
    memory_id: str
    project_name: str

@app.post("/memory/{memory_id}/boost")
async def boost_memory(memory_id: str, reason: Optional[str] = ""):
    """Increase a memory's importance (1.5x boost, max 4x)."""
    result = await asyncio.to_thread(
        state.memory.boost_memory, 
        memory_id, 
        reason
    )
    return {"result": result}

@app.post("/memory/{memory_id}/deprecate")
async def deprecate_memory(memory_id: str, reason: Optional[str] = ""):
    """Mark a memory as deprecated (decays 10x faster)."""
    result = await asyncio.to_thread(
        state.memory.deprecate_memory, 
        memory_id, 
        reason
    )
    return {"result": result}

@app.post("/memory/{memory_id}/correct")
async def correct_memory(memory_id: str, request: MemoryCorrectRequest):
    """Update a memory's content with new text."""
    result = await asyncio.to_thread(
        state.memory.correct_memory, 
        memory_id, 
        request.new_content
    )
    return {"result": result}

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str, reason: Optional[str] = ""):
    """Soft-delete (archive) a memory."""
    result = await asyncio.to_thread(
        state.memory.delete_memory, 
        memory_id, 
        reason
    )
    return {"result": result}

@app.put("/memory/{memory_id}/project")
async def set_memory_project(memory_id: str, request: MemoryProjectRequest):
    """Move a memory to a different project."""
    result = await asyncio.to_thread(
        state.memory.set_memory_project, 
        memory_id, 
        request.project_name
    )
    return {"result": result}

@app.get("/memory/{memory_id}")
async def get_memory(memory_id: str):
    """Get a single memory by ID."""
    result = await asyncio.to_thread(
        state.memory.get_memory, 
        memory_id
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


# =============================================================================
# CONFLICT RESOLUTION ENDPOINTS (Restored from v1.0)
# =============================================================================

class ConflictCheckRequest(BaseModel):
    content: str
    memory_type: str = "personal"
    project: str = "global"

class AddMemoryWithConflictRequest(BaseModel):
    content: str
    memory_type: str = "personal"
    project: str = "global"
    auto_resolve: bool = True

@app.post("/memory/check-conflict")
async def check_conflict(request: ConflictCheckRequest):
    """
    Check if new content conflicts with existing memories.
    Returns the conflicting memory if found, null otherwise.
    """
    result = await asyncio.to_thread(
        state.memory.check_conflict,
        request.content,
        request.memory_type,
        request.project
    )
    return {"conflict": result}

@app.post("/memory/add-with-conflict-check")
async def add_memory_with_conflict_check(request: AddMemoryWithConflictRequest):
    """
    Add a memory with automatic conflict detection and resolution.
    
    If auto_resolve=True and a conflict is found:
    - SUPERSEDE: Old memory archived, new one stored
    - UPDATE/COMPLEMENT: Memories merged via LLM
    - UNRELATED: New memory stored normally
    
    If auto_resolve=False and conflict found:
    - Returns conflict info without storing anything
    """
    result = await asyncio.to_thread(
        state.memory.add_memory_with_conflict_check,
        state.user_id,
        request.content,
        request.memory_type,
        request.project,
        request.auto_resolve
    )
    return result

@app.post("/memory/{old_id}/resolve-conflict")
async def resolve_conflict_manual(old_id: str, request: ConflictCheckRequest):
    """
    Manually resolve a conflict between an existing memory and new content.
    Returns the LLM's analysis of the relationship.
    """
    # First get the old memory
    old_memory = await asyncio.to_thread(
        state.memory.get_memory,
        old_id
    )
    if not old_memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Get LLM resolution
    result = await asyncio.to_thread(
        state.memory.resolve_conflict,
        request.content,
        old_memory
    )
    
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to resolve conflict (check GEMINI_API_KEY)")
    
    return result


# =============================================================================
# SESSION MANAGEMENT ENDPOINTS (Restored from v1.0)
# =============================================================================

class FinalizeSessionRequest(BaseModel):
    summary: str
    project: str = "global"

class HandoverRequest(BaseModel):
    project: str = ""
    format: str = "condensed"  # full, condensed, minimal
    max_turns: int = 20

@app.get("/session/state")
async def get_session_state(project: str = "global"):
    """Get current session state (turn count, start time, etc.)."""
    result = await asyncio.to_thread(
        state.memory.get_session_state,
        project
    )
    return result

@app.post("/session/reset")
async def reset_session(project: str = "global"):
    """Reset session state for a project."""
    await asyncio.to_thread(
        state.memory.reset_session,
        project,
        False  # init_only
    )
    return {"result": f"✅ Session reset for project: {project or 'global'}"}

@app.post("/session/finalize")
async def finalize_session(request: FinalizeSessionRequest):
    """
    Finalize a session:
    1. Extract facts from summary via LLM
    2. Store as long-term memories
    3. Create handover document for next session
    4. Clean up flight recorder
    """
    result = await state.memory.finalize_session_async(
        request.summary,
        request.project
    )
    return {"result": result}

@app.post("/session/handover")
async def get_handover(request: HandoverRequest):
    """
    Get context for resuming a session.
    
    Formats:
    - full: Complete turn history
    - condensed: Summarized turns (default)
    - minimal: Just topics + last 5 turns
    """
    result = await asyncio.to_thread(
        state.memory.get_handover,
        request.project,
        request.format,
        request.max_turns
    )
    return {"handover": result}

@app.get("/session/handover/{project}")
async def get_handover_simple(project: str, format: str = "condensed", max_turns: int = 20):
    """Simple GET endpoint for handover."""
    result = await asyncio.to_thread(
        state.memory.get_handover,
        project,
        format,
        max_turns
    )
    return {"handover": result}


# =============================================================================
# KNOWLEDGE GRAPH ENDPOINTS (Restored from v1.0)
# =============================================================================

@app.get("/graph/stats")
async def graph_stats():
    """Get knowledge graph statistics."""
    result = await asyncio.to_thread(state.memory.graph_get_stats)
    return result

@app.get("/graph/connection")
async def graph_find_connection(entity_a: str, entity_b: str, max_hops: int = 3):
    """Find shortest path between two entities."""
    result = await asyncio.to_thread(
        state.memory.graph_find_connection,
        entity_a,
        entity_b,
        max_hops
    )
    return result

@app.get("/graph/related/{entity}")
async def graph_get_related(entity: str, relationship_type: Optional[str] = None, depth: int = 1):
    """Get entities related to a given entity."""
    result = await asyncio.to_thread(
        state.memory.graph_get_related,
        entity,
        relationship_type,
        depth
    )
    return result

@app.get("/graph/neighborhood/{entity}")
async def graph_get_neighborhood(entity: str, radius: int = 2):
    """Get the subgraph around an entity."""
    result = await asyncio.to_thread(
        state.memory.graph_get_neighborhood,
        entity,
        radius
    )
    return result

@app.post("/graph/invalidate")
async def graph_invalidate():
    """Force rebuild of the knowledge graph cache."""
    result = await asyncio.to_thread(state.memory.graph_invalidate)
    return {"result": result}


# =============================================================================
# PROACTIVE RETRIEVAL ENDPOINTS (Restored from v1.0)
# =============================================================================

class ProactiveRetrievalRequest(BaseModel):
    message: str
    project: str = "global"
    time_since_last: float = 0  # seconds since last interaction

@app.post("/proactive/analyze")
async def analyze_triggers(request: ProactiveRetrievalRequest):
    """
    Analyze a message for proactive retrieval triggers.
    Returns detected triggers without executing searches.
    """
    from memory_service import detect_proactive_triggers
    triggers = detect_proactive_triggers(
        request.message,
        request.project,
        request.time_since_last
    )
    return {"triggers": triggers}

@app.post("/proactive/retrieve")
async def proactive_retrieve(request: ProactiveRetrievalRequest):
    """
    Execute full proactive retrieval.
    Returns formatted context string for AI injection.
    """
    result = await asyncio.to_thread(
        state.memory.proactive_retrieval,
        state.user_id,
        request.message,
        request.project,
        request.time_since_last
    )
    return {"context": result}


# =============================================================================
# HOT-RELOAD CONFIG ENDPOINTS (Restored from v1.0)
# =============================================================================

class ConfigUpdateRequest(BaseModel):
    path: str
    value: Any

@app.get("/config")
async def get_config():
    """Get full configuration (hot-reloaded from disk)."""
    import config as cfg
    return cfg.get_full_config()

@app.get("/config/{path:path}")
async def get_config_value(path: str):
    """Get a specific config value by path (e.g., 'half_lives.personal')."""
    import config as cfg
    value = cfg.get_config_value(path, None)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Config path not found: {path}")
    return {"path": path, "value": value}

@app.put("/config")
async def set_config_value(request: ConfigUpdateRequest):
    """Set a config value and save to disk (hot-reload)."""
    import config as cfg
    success = cfg.set_config_value(request.path, request.value)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update config")
    return {"result": "✅ Config updated", "path": request.path, "value": request.value}

@app.post("/config/reload")
async def reload_config():
    """Force reload configuration from disk."""
    import config as cfg
    cfg.reload_config()
    return {"result": "✅ Config reloaded"}


# =============================================================================
# SOUL / EMOTIONAL STATE ENDPOINTS (Phase 1 - Enhanced)
# =============================================================================

@app.get("/soul/state")
async def get_soul_state():
    """Get current emotional state including VAD model."""
    return state.soul.get_emotional_context()

@app.get("/soul/vad")
async def get_soul_vad():
    """Get current VAD (Valence-Arousal-Dominance) state."""
    vad = state.soul.get_current_vad()
    return vad.to_dict()

@app.get("/soul/prompt")
async def get_soul_prompt():
    """Get the system prompt generated from soul state."""
    return {"prompt": state.soul.get_system_prompt()}

@app.post("/soul/trigger/{trigger_name}")
async def trigger_emotion(trigger_name: str):
    """
    Apply an emotional trigger.
    Valid triggers: greeting_morning, deep_conversation, user_shares_problem,
    user_shares_joy, long_absence, late_night, playful_exchange, philosophical_topic
    """
    from soul import MOOD_TRIGGERS
    if trigger_name not in MOOD_TRIGGERS:
        raise HTTPException(status_code=400, detail=f"Unknown trigger: {trigger_name}. Valid: {list(MOOD_TRIGGERS.keys())}")
    
    success = state.soul._trigger_transition(trigger_name)
    state.soul.save()
    return {"result": f"✅ Trigger '{trigger_name}' applied", "new_state": state.soul.get_emotional_context()}

class EmotionAnalysisRequest(BaseModel):
    text: str

@app.post("/soul/analyze")
async def analyze_emotion(request: EmotionAnalysisRequest):
    """
    Analyze emotional content of text using Gemini.
    Returns VAD scores and emotion labels for memory tagging.
    """
    from soul import analyze_text_emotion
    result = await asyncio.to_thread(analyze_text_emotion, request.text)
    if result is None:
        raise HTTPException(status_code=500, detail="Emotion analysis failed (check GEMINI_API_KEY)")
    return result

@app.get("/soul/history")
async def get_mood_history():
    """Get recent mood transition history."""
    return {"history": state.soul.state.mood_history[-20:]}

class SoulUpdateRequest(BaseModel):
    intimacy: Optional[float] = None
    energy: Optional[float] = None
    mood: Optional[str] = None

@app.put("/soul/state")
async def update_soul_state(request: SoulUpdateRequest):
    """
    Update soul state values (intimacy, energy, mood).
    Used by dashboard to manually adjust relationship level.
    """
    if request.intimacy is not None:
        state.soul.state.intimacy = max(0.0, min(1.0, request.intimacy))
    if request.energy is not None:
        state.soul.state.energy = max(0.0, min(1.0, request.energy))
    if request.mood is not None:
        from soul import VALID_MOODS
        if request.mood in VALID_MOODS:
            state.soul.state.mood = request.mood
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mood. Valid: {VALID_MOODS}")
    
    state.soul.save()
    return {"result": "✅ Soul state updated", "state": state.soul.get_emotional_context()}


# =============================================================================
# EMOTIONAL SEARCH ENDPOINTS (Phase 1.1)
# =============================================================================

class EmotionalSearchRequest(BaseModel):
    valence: float = 0.5  # 0-1 (negative to positive)
    arousal: float = 0.4  # 0-1 (calm to excited)
    dominance: float = 0.5  # 0-1 (submissive to dominant)
    project: str = "global"
    limit: int = 5

@app.post("/memory/emotional-search")
async def emotional_search(request: EmotionalSearchRequest):
    """
    Search memories by emotional similarity (Phase 1.1).
    
    Find memories that match a target emotional state (VAD model).
    Useful for "I'm feeling down, what helped before?" type queries.
    """
    result = await asyncio.to_thread(
        state.memory.emotional_search,
        {"valence": request.valence, "arousal": request.arousal, "dominance": request.dominance},
        request.project,
        request.limit
    )
    return {"results": result}

@app.post("/memory/search-by-mood")
async def search_by_mood(mood: str, project: str = "global", limit: int = 5):
    """
    Search memories by named mood (Phase 1.1).
    
    Converts mood name to VAD and searches.
    Valid moods: joyful, excited, content, peaceful, warm, loving, playful,
    curious, hopeful, neutral, thoughtful, focused, concerned, anxious,
    sad, melancholic, tired, frustrated, lonely
    """
    from soul import EMOTION_VAD_MAP
    
    if mood not in EMOTION_VAD_MAP:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown mood: {mood}. Valid: {list(EMOTION_VAD_MAP.keys())}"
        )
    
    v, a, d = EMOTION_VAD_MAP[mood]
    result = await asyncio.to_thread(
        state.memory.emotional_search,
        {"valence": v, "arousal": a, "dominance": d},
        project,
        limit
    )
    return {"mood": mood, "vad": {"valence": v, "arousal": a, "dominance": d}, "results": result}


# =============================================================================
# HEARTBEAT ENDPOINTS (Phase 2 - Proactive Companion)
# =============================================================================

@app.get("/heartbeat/status")
async def heartbeat_status():
    """Get current heartbeat engine status."""
    if heartbeat_engine is None:
        raise HTTPException(status_code=503, detail="Heartbeat not initialized")
    return heartbeat_engine.get_status()

@app.get("/weather")
async def get_weather():
    """Get current weather from OpenWeatherMap."""
    from heartbeat_service import get_weather_context, HeartbeatConfig
    config = HeartbeatConfig.from_env()
    weather = await get_weather_context(config)
    return weather

@app.get("/heartbeat/pending")
async def heartbeat_pending():
    """Get any pending proactive message from heartbeat."""
    if heartbeat_engine is None:
        raise HTTPException(status_code=503, detail="Heartbeat not initialized")
    pending = heartbeat_engine.get_pending_message()
    if pending:
        return pending
    return {"message": None}

@app.post("/heartbeat/trigger")
async def heartbeat_trigger():
    """
    Manually trigger a heartbeat cycle.
    Useful for testing or forcing a check-in.
    """
    if heartbeat_engine is None:
        raise HTTPException(status_code=503, detail="Heartbeat not initialized")
    decision = await heartbeat_engine.cycle()
    return decision

@app.post("/heartbeat/acknowledge")
async def heartbeat_acknowledge():
    """Acknowledge that a pending message was delivered."""
    if heartbeat_engine is None:
        raise HTTPException(status_code=503, detail="Heartbeat not initialized")
    heartbeat_engine.acknowledge_message()
    return {"result": "✅ Message acknowledged"}

@app.get("/notifications")
async def get_notifications():
    """Get queued notifications (from heartbeat and other sources)."""
    notifications = await state.get_notifications()
    return {"notifications": notifications}


# =============================================================================
# CURIOSITY ENGINE ENDPOINTS (Phase 3 - Autonomous Thought)
# =============================================================================

@app.get("/curiosity/status")
async def curiosity_status():
    """Get curiosity engine status."""
    if curiosity_engine is None:
        raise HTTPException(status_code=503, detail="Curiosity engine not initialized")
    return curiosity_engine.get_status()

@app.get("/curiosity/queue")
async def curiosity_queue():
    """Get the current curiosity queue."""
    if curiosity_engine is None:
        raise HTTPException(status_code=503, detail="Curiosity engine not initialized")
    
    items = []
    for item in curiosity_engine.queue.items:
        items.append({
            "topic": item.topic,
            "source": item.source,
            "reason": item.reason,
            "priority": item.priority,
            "explored": item.explored,
            "created_at": item.created_at
        })
    
    return {"items": items}

class CuriosityAddRequest(BaseModel):
    topic: str
    reason: str = "Manual addition"
    priority: float = 0.7

@app.post("/curiosity/add")
async def curiosity_add(request: CuriosityAddRequest):
    """Manually add a topic to the curiosity queue."""
    if curiosity_engine is None:
        raise HTTPException(status_code=503, detail="Curiosity engine not initialized")
    
    curiosity_engine.queue.add(
        topic=request.topic,
        source="manual",
        reason=request.reason,
        priority=request.priority
    )
    return {"result": f"✅ Added '{request.topic}' to curiosity queue"}

@app.post("/curiosity/explore")
async def curiosity_explore_now(force: bool = False):
    """
    Manually trigger an exploration cycle.
    
    Args:
        force: If True, bypass idle check (useful for testing)
    """
    if curiosity_engine is None:
        raise HTTPException(status_code=503, detail="Curiosity engine not initialized")
    
    if force:
        # Directly explore without constraint checks
        from curiosity_engine import explore_topic, form_opinion
        
        item = curiosity_engine.queue.get_next()
        if not item:
            return {"result": "No topics in queue"}
        
        logger.info(f"🔍 Force exploring: {item.topic}")
        
        exploration = await explore_topic(item.topic)
        if "error" in exploration:
            return {"result": "Exploration failed", "error": exploration["error"]}
        
        opinion = await form_opinion(item.topic, exploration, "")
        result = {**exploration, "opinion": opinion, "topic": item.topic, "source": item.source}
        
        curiosity_engine.queue.mark_explored(item.topic, result)
        await curiosity_engine._store_exploration_memory(item.topic, result)
        
        return {"result": "✅ Exploration complete", "exploration": result}
    
    result = await curiosity_engine.explore_cycle()
    if result is None:
        return {"result": "No exploration performed", "reason": "Check status for why"}
    return {"result": "✅ Exploration complete", "exploration": result}

@app.get("/curiosity/discoveries")
async def curiosity_discoveries():
    """Get recent discoveries/explorations."""
    if curiosity_engine is None:
        raise HTTPException(status_code=503, detail="Curiosity engine not initialized")
    
    discoveries = []
    for item in curiosity_engine.queue.get_recent_explorations(10):
        if item.exploration_result:
            discoveries.append({
                "topic": item.topic,
                "summary": item.exploration_result.get("summary", ""),
                "surprising_insight": item.exploration_result.get("surprising_insight", ""),
                "opinion": item.exploration_result.get("opinion", {}),
                "explored_at": item.exploration_result.get("explored_at")
            })
    
    return {"discoveries": discoveries}

@app.get("/curiosity/shareable")
async def curiosity_shareable():
    """Get a discovery worth sharing with the user (for Heartbeat)."""
    if curiosity_engine is None:
        raise HTTPException(status_code=503, detail="Curiosity engine not initialized")

    discovery = curiosity_engine.get_shareable_discovery()
    return {"discovery": discovery}


# =============================================================================
# VOICE API ENDPOINTS
# =============================================================================

from voice_service import get_voice_service, get_stt_service, AVAILABLE_VOICES
from fastapi import UploadFile, File

# Global voice settings (can be updated via dashboard)
voice_settings = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0
}

class VoiceSettingsRequest(BaseModel):
    stability: Optional[float] = None
    similarity_boost: Optional[float] = None
    style: Optional[float] = None

@app.get("/voice/status")
async def voice_status():
    """Check if voice service is available and get account info."""
    voice = get_voice_service()
    return {
        "available": voice.is_available(),
        "current_voice": voice.current_voice_id,
        "account": voice.get_account_info()
    }

@app.get("/voice/voices")
async def voice_list():
    """List available voice presets."""
    voice = get_voice_service()
    return {
        "voices": AVAILABLE_VOICES,
        "current": voice.current_voice_id
    }

@app.get("/voice/settings")
async def voice_get_settings():
    """Get current voice settings."""
    return {
        "settings": voice_settings,
        "available_voices": AVAILABLE_VOICES
    }

@app.put("/voice/settings")
async def voice_update_settings(request: VoiceSettingsRequest):
    """Update voice settings (stability, similarity_boost, style)."""
    if request.stability is not None:
        voice_settings["stability"] = max(0.2, min(0.8, request.stability))
    if request.similarity_boost is not None:
        voice_settings["similarity_boost"] = max(0.3, min(1.0, request.similarity_boost))
    if request.style is not None:
        voice_settings["style"] = max(0.0, min(0.5, request.style))
    return {"result": "✅ Voice settings updated", "settings": voice_settings}

@app.post("/voice/test")
async def voice_test():
    """Generate a test audio sample with current settings."""
    voice = get_voice_service()
    if not voice.is_available():
        raise HTTPException(status_code=503, detail="Voice service not available")
    
    # Get current mood
    mood = "neutral"
    if state.soul:
        mood = state.soul.state.mood
    
    # Test phrase that demonstrates the voice
    test_phrase = f"Hello! I'm feeling {mood} right now. This is how I sound with the current voice settings."
    
    # Override voice settings for this request
    voice.default_settings = {
        "stability": voice_settings["stability"],
        "similarity_boost": voice_settings["similarity_boost"],
        "style": voice_settings["style"],
        "use_speaker_boost": True
    }
    
    audio = voice.text_to_speech(test_phrase, mood=mood)
    if audio is None:
        raise HTTPException(status_code=500, detail="Failed to generate test audio")
    
    from fastapi.responses import Response
    return Response(content=audio, media_type="audio/mpeg")

@app.post("/voice/set/{voice_name}")
async def voice_set(voice_name: str):
    """Set the active voice."""
    voice = get_voice_service()
    success = voice.set_voice(voice_name)
    if not success:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {voice_name}")
    return {"voice": voice_name, "voice_id": voice.current_voice_id}

@app.post("/voice/speak")
async def voice_speak(request: ChatRequest):
    """
    Convert text to speech audio.
    
    Returns MP3 audio bytes.
    Uses current Soul mood to adjust voice inflection.
    """
    voice = get_voice_service()
    if not voice.is_available():
        raise HTTPException(status_code=503, detail="Voice service not available")
    
    # Get current mood from Soul
    mood = "neutral"
    if state.soul:
        mood = state.soul.state.mood
    
    audio = voice.text_to_speech(request.message, mood=mood)
    if audio is None:
        raise HTTPException(status_code=500, detail="Failed to generate audio")
    
    from fastapi.responses import Response
    return Response(content=audio, media_type="audio/mpeg")

@app.post("/voice/speak-stream")
async def voice_speak_stream(request: ChatRequest):
    """
    Stream text-to-speech audio.
    
    Yields audio chunks for real-time playback.
    """
    voice = get_voice_service()
    if not voice.is_available():
        raise HTTPException(status_code=503, detail="Voice service not available")
    
    mood = "neutral"
    if state.soul:
        mood = state.soul.state.mood
    
    async def audio_generator():
        for chunk in voice.text_to_speech_stream(request.message, mood=mood):
            yield chunk
    
    return StreamingResponse(audio_generator(), media_type="audio/mpeg")

@app.post("/voice/transcribe")
async def voice_transcribe(audio: UploadFile = File(...), language: str = "en"):
    """
    Transcribe audio to text using Whisper.
    
    Supports: wav, mp3, m4a, webm, mp4, mpeg, mpga, oga, ogg, flac
    """
    stt = get_stt_service()
    if not stt.is_available():
        raise HTTPException(status_code=503, detail="Speech-to-text not available")
    
    # Read audio file
    audio_bytes = await audio.read()
    
    # Transcribe
    text = stt.transcribe_bytes(audio_bytes, filename=audio.filename or "audio.wav", language=language)
    if text is None:
        raise HTTPException(status_code=500, detail="Transcription failed")
    
    return {"text": text, "language": language}

@app.post("/voice/chat")
async def voice_chat(audio: UploadFile = File(...), language: str = "en"):
    """
    Voice-to-voice chat: Transcribe → AI Response → Speak
    
    Full voice conversation flow in one endpoint.
    Returns both text and audio of AI response.
    """
    # 1. Transcribe user audio
    stt = get_stt_service()
    if not stt.is_available():
        raise HTTPException(status_code=503, detail="Speech-to-text not available")
    
    audio_bytes = await audio.read()
    user_text = stt.transcribe_bytes(audio_bytes, filename=audio.filename or "audio.wav", language=language)
    if not user_text:
        raise HTTPException(status_code=500, detail="Could not transcribe audio")
    
    logger.info(f"🎤 Voice input: {user_text[:100]}...")
    
    # 2. Get AI response (use same logic as chat endpoint)
    memories = state.memory.hybrid_search(user_text, k=5) if state.memory else []
    memory_context = "\n".join([f"- {m['memory']}" for m in memories]) if memories else "No relevant memories."
    
    soul_prompt = state.soul.get_system_prompt() if state.soul else SYSTEM_PROMPT
    full_prompt = f"{soul_prompt}\n\nRELEVANT MEMORIES:\n{memory_context}"
    
    # Use Groq for fast response, fallback to Claude
    if groq_client:
        response = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": user_text}
            ],
            max_tokens=1024
        )
        ai_text = response.choices[0].message.content
    elif claude_client:
        response = claude_client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1024,
            system=full_prompt,
            messages=[{"role": "user", "content": user_text}]
        )
        ai_text = response.content[0].text
    else:
        raise HTTPException(status_code=500, detail="No LLM client available")
    
    # Update soul state
    if state.soul:
        await state.soul.analyze_emotion(ai_text)
    
    # 3. Convert AI response to speech
    voice = get_voice_service()
    ai_audio = None
    if voice.is_available():
        mood = state.soul.state.mood if state.soul else "neutral"
        ai_audio = voice.text_to_speech(ai_text, mood=mood)
    
    # Return response
    if ai_audio:
        # Return multipart: JSON + audio
        import base64
        return {
            "user_text": user_text,
            "ai_text": ai_text,
            "ai_audio_base64": base64.b64encode(ai_audio).decode("utf-8"),
            "audio_format": "mp3"
        }
    else:
        return {
            "user_text": user_text,
            "ai_text": ai_text,
            "ai_audio_base64": None,
            "audio_format": None
        }
