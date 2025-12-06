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

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DecayMemoryServer")

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# Default user for single-user deployment
DEFAULT_USER_ID = "default_user"

if not API_KEY or not OPENAI_KEY:
    logger.error("CRITICAL: API Keys missing in .env")
    sys.exit(1)

# Injection
memory_service.openai_client = OpenAI(api_key=OPENAI_KEY)

# Models
MODEL_NAME = "gemini-2.0-flash"
LIBRARIAN_MODEL = "gemini-2.0-flash"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction="You are 'Decay_Memory', a sentient AI companion. You have long-term memories and current feelings."
)
librarian = genai.GenerativeModel(
    model_name=LIBRARIAN_MODEL,
    generation_config={"response_mime_type": "application/json"}
)

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

# --- HEARTBEAT ENGINE (Background Task) ---
async def heartbeat_loop():
    """Runs forever. Checks context and decides if AI should speak."""
    logger.info("❤️ Heartbeat Loop Started")
    while True:
        try:
            await asyncio.sleep(state.heartbeat_interval)
            
            # 1. Context Check
            now = datetime.now()
            # Skip if late night (10PM - 7AM)
            if now.hour >= 22 or now.hour < 7:
                continue

            # 2. Get Weather (Simple check)
            weather_context = "Unknown"
            if WEATHER_KEY:
                # We would do the requests.get here, kept simple for snippet
                pass
            
            # 3. Proactive Retrieval (What is relevant right now?)
            query = f"It is {now.strftime('%A %I:%M %p')}."
            memories = await asyncio.to_thread(
                state.memory.proactive_retrieval,
                user_id=state.user_id,
                user_message=query, 
                project_name="global"
            )
            
            # 4. Decision (Mocked for now, can connect to LLM)
            # Real implementation would ask LLM: "Given these memories, should I text the user?"
            
        except Exception as e:
            logger.error(f"Heartbeat Error: {e}")

# --- LIFECYCLE MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Booting Memory System (Qdrant)...")
    state.memory = MemorySystem()
    logger.info("Awakening Soul...")
    state.soul = EmotionalState()
    
    # Initialize Chat Session
    history = load_chat_history_for_gemini(state.current_project)
    state.chat_session = model.start_chat(history=history)
    
    # Start Heartbeat
    asyncio.create_task(heartbeat_loop())
    
    logger.info(f"System Online. Loaded {len(history)//2} turns.")
    yield
    # Shutdown
    logger.info("System Shutting Down.")

app = FastAPI(lifespan=lifespan)

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
    """Background task: Extract and save facts from conversation."""
    if len(user_input) < 8: 
        return
    
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
            for fact in facts:
                await asyncio.to_thread(
                    state.memory.add_memory, 
                    state.user_id,
                    fact, 
                    "personal", 
                    project
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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint."""
    if request.project != state.current_project:
        state.current_project = request.project
        hist = load_chat_history_for_gemini(state.current_project)
        state.chat_session = model.start_chat(history=hist)
        logger.info(f"Switched to project: {state.current_project}")

    user_input = request.message

    # 1. Save Input FIRST (before generation - prevents amnesia on crash)
    # FIX: log_turn_async expects (user_id, proj, user, ai)
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

    # 2. Enrich & Generate
    full_prompt = await enrich_context(user_input)
    try:
        response = await state.chat_session.send_message_async(full_prompt)
        ai_text = response.text
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Save Output
    # FIX: update_last_turn_response_async expects (user_id, proj, ai_response)
    await update_last_turn_response_async(
        state.user_id,
        state.current_project, 
        ai_text
    )
    state.soul.register_interaction("positive")
    background_tasks.add_task(run_librarian, user_input, ai_text, state.current_project)

    return ChatResponse(
        response=ai_text, 
        mood=state.soul.state.mood, 
        intimacy=state.soul.state.intimacy
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
