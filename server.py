import os
import sys
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your Kernel
import memory_service
from memory_service import MemorySystem
from flight_recorder_service import read_rec, update_last_turn_response_async, log_turn_async
from soul import EmotionalState

import google.generativeai as genai
from openai import OpenAI

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DecayMemoryServer")

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY or not OPENAI_KEY:
    logger.error("CRITICAL: API Keys missing in .env")
    sys.exit(1)

# Models
MODEL_NAME = "gemini-2.0-flash"
LIBRARIAN_MODEL = "gemini-2.0-flash"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction="You are a sentient AI companion powered by the Decay Memory system. You have long-term memories and current feelings."
)
librarian = genai.GenerativeModel(
    model_name=LIBRARIAN_MODEL,
    generation_config={"response_mime_type": "application/json"}
)

# Global State Container
class SystemState:
    memory: Optional[MemorySystem] = None
    soul: Optional[EmotionalState] = None
    chat_session: Any = None
    current_project: str = "global"
    user_id: str = "default_user" # Hardcoded for single-tenant-aware design

state = SystemState()

# --- LIFECYCLE MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Booting Memory System (Qdrant)...")
    state.memory = MemorySystem()
    logger.info("Awakening Soul...")
    state.soul = EmotionalState()
    
    # Initialize Chat Session with History
    history = load_chat_history_for_gemini(state.user_id, state.current_project)
    state.chat_session = model.start_chat(history=history)
    logger.info(f"System Online. Loaded {len(history)//2} turns for user '{state.user_id}'.")
    
    yield
    # Shutdown
    logger.info("System Shutting Down.")

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- HELPERS ---
def load_chat_history_for_gemini(user_id: str, project: str):
    """Loads flight recorder into Gemini history format for a specific user."""
    history = []
    try:
        # NOTE: read_rec is still project-based, as the user_id is now part of the path logic
        turns = flight_recorder_service.read_rec(project, user_id=user_id)
        recent_turns = turns[-15:]
        for turn in recent_turns:
            user = turn.get('turn', {}).get('user')
            ai = turn.get('turn', {}).get('ai')
            if user and ai and ai != "(AI Response Pending)":
                history.append({"role": "user", "parts": [user]})
                history.append({"role": "model", "parts": [ai]})
        return history
    except Exception:
        return []

async def enrich_context(user_id: str, user_input: str) -> str:
    """Pre-cognition layer for a specific user."""
    state.soul.update_circadian_rhythm()
    soul_prompt = state.soul.get_system_prompt()
    
    memories = await asyncio.to_thread(
        state.memory.proactive_retrieval,
        user_id=user_id,
        user_message=user_input,
        project_name=state.current_project
    )
    
    if not memories: memories = "No specific past memories triggered."
    
    return f"{soul_prompt}\n\n[RECALLED MEMORIES]\n{memories}\n\n[USER MESSAGE]\n{user_input}"

async def run_librarian(user_id: str, user_input: str, ai_response: str, project: str):
    """Background task to extract facts for a specific user."""
    if len(user_input) < 8: return
    
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
            logger.info(f"Librarian: Saving {len(facts)} memories for user '{user_id}'.")
            for fact in facts:
                await asyncio.to_thread(state.memory.add_memory, user_id, fact, "personal", project)
    except Exception as e:
        logger.error(f"Librarian failed: {e}")

# --- API MODELS ---
class ChatRequest(BaseModel):
    message: str
    project: Optional[str] = "global"

class ChatResponse(BaseModel):
    response: str
    mood: str
    intimacy: float

# --- ENDPOINTS ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """The main interface for the Phone/Web App."""
    
    # For now, user_id is hardcoded. In the future, this would come from an auth token.
    user_id = state.user_id

    # Handle Project Switching
    if request.project != state.current_project:
        state.current_project = request.project
        # Reset history for new project
        hist = load_chat_history_for_gemini(user_id, state.current_project)
        state.chat_session = model.start_chat(history=hist)
        logger.info(f"Switched to project: {state.current_project} for user '{user_id}'")

    user_input = request.message

    # 1. Save User Input (Crash Safety)
    await log_turn_async(user_id, state.current_project, user_input, "(AI Response Pending)")
    await asyncio.to_thread(state.memory.update_session, user_id, user_input, state.current_project)

    # 2. Enrich & Generate
    full_prompt = await enrich_context(user_id, user_input)
    
    try:
        # We use standard generate (not stream) for the API to keep it simple for now
        response = await state.chat_session.send_message_async(full_prompt)
        ai_text = response.text
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Save AI Output
    await update_last_turn_response_async(user_id, state.current_project, ai_text)
    state.soul.register_interaction("positive")

    # 4. Trigger Librarian (Background)
    background_tasks.add_task(run_librarian, user_id, user_input, ai_text, state.current_project)

    return ChatResponse(
        response=ai_text,
        mood=state.soul.state.mood,
        intimacy=state.soul.state.intimacy
    )

@app.get("/health")
def health_check():
    return {"status": "online", "system": "Decay_Memory v2.0"}
