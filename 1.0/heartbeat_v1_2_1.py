"""
Heartbeat Service for AI Companion - V1.2.1 (Fixed Imports)

Changes:
- FIXED: Import paths now point correctly to 'memory_service.py' instead of non-existent 'decay_memory' package.
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import re
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from dotenv import load_dotenv
load_dotenv()

# --- IMPORT FIX IS HERE ---
try:
    # We import directly from the file 'memory_service.py'
    import memory_service
    from memory_service import MemorySystem
    from memory_service import read_rec, log_turn_async
    from memory_service import FLIGHT_DIR
except ImportError as e:
    print(f"CRITICAL: Could not import memory_service.py. Error: {e}")
    print("Ensure 'memory_service.py' is in the same folder as this script.")
    sys.exit(1)

# =============================================================================
# SECURITY UTILS
# =============================================================================

def mask_key(key: str) -> str:
    """Mask API keys for logging."""
    if not key or len(key) < 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"

def sanitize_text(text: str, limit: int = 1000) -> str:
    """
    Sanitize text to prevent basic prompt injection or control char exploits.
    """
    if not text: return ""
    text = " ".join(text.split())
    text = "".join(ch for ch in text if ch.isprintable())
    return text[:limit]

def atomic_write(filepath: Path, data: Dict[str, Any]) -> None:
    """Write JSON to file atomically."""
    dir_name = filepath.parent
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8') as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name
    
    try:
        shutil.move(temp_name, filepath)
    except Exception as e:
        os.remove(temp_name)
        raise e

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HeartbeatConfig:
    """Configuration with validation."""
    # Timing
    interval_seconds: int = 300
    min_gap_seconds: int = 3600
    quiet_hours_start: int = 22
    quiet_hours_end: int = 7
    
    # Limits
    max_daily_messages: int = 5
    max_memory_length: int = 2500
    max_message_length: int = 500
    
    # Location
    location: str = "Arroyo Grande, CA"
    latitude: float = 35.1186
    longitude: float = -120.5910
    
    # Weather
    weather_api_key: str = ""
    
    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    decision_threshold: float = 0.7
    request_timeout: int = 15
    
    # Personality
    companion_name: str = "Her"
    
    # Files
    state_file: str = "heartbeat_state.json"
    mood_file: str = "heartbeat_mood.json"

    def validate(self):
        if self.interval_seconds < 60:
            logging.warning("Interval < 60s is aggressive. Setting to 60s.")
            self.interval_seconds = 60
        
        valid_providers = ["openai", "gemini"]
        if self.llm_provider not in valid_providers:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")

# Load and validate
CONFIG = HeartbeatConfig(
    weather_api_key=os.getenv("OPENWEATHERMAP_API_KEY", ""),
    llm_provider=os.getenv("HEARTBEAT_LLM_PROVIDER", "openai"),
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [HEARTBEAT] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Heartbeat")

# =============================================================================
# MOOD STATE
# =============================================================================

@dataclass
class MoodState:
    current_mood: str = "neutral"
    energy_level: float = 0.7
    intimacy_level: float = 0.75
    last_interaction_sentiment: str = "positive"
    circadian_phase: str = "day"
    consecutive_no_response: int = 0
    last_update: float = 0.0
    
    def decay(self) -> None:
        hours_since_update = (time.time() - self.last_update) / 3600
        hour = datetime.now().hour
        
        if 6 <= hour < 12:
            self.circadian_phase = "morning"
            self.energy_level = min(1.0, self.energy_level + 0.1)
        elif 12 <= hour < 18:
            self.circadian_phase = "day"
        elif 18 <= hour < 22:
            self.circadian_phase = "evening"
            self.energy_level = max(0.3, self.energy_level - 0.05)
        else:
            self.circadian_phase = "night"
            self.energy_level = max(0.2, self.energy_level - 0.1)
            
        if hours_since_update > 24:
            self.current_mood = "neutral"
            
        if hours_since_update > 48:
            self.intimacy_level = max(0.5, self.intimacy_level - 0.05)

        self.last_update = time.time()

class MoodManager:
    def __init__(self, mood_file: str = "heartbeat_mood.json"):
        self.mood_file = Path(mood_file)
        self.state = self._load()
    
    def _load(self) -> MoodState:
        if self.mood_file.exists():
            try:
                data = json.loads(self.mood_file.read_text(encoding='utf-8'))
                return MoodState(**data)
            except Exception as e:
                logger.warning(f"Corrupt mood file, resetting: {e}")
        return MoodState(last_update=time.time())
    
    def save(self) -> None:
        try:
            atomic_write(self.mood_file, asdict(self.state))
        except Exception as e:
            logger.error(f"Failed to save mood state: {e}")
    
    def update(self) -> MoodState:
        self.state.decay()
        self.save()
        return self.state

# =============================================================================
# CONTEXT GATHERERS
# =============================================================================

def get_time_context() -> Dict[str, Any]:
    now = datetime.now()
    return {
        "datetime": now.isoformat(),
        "day_of_week": now.strftime("%A"),
        "time_of_day": now.strftime("%I:%M %p"),
        "hour": now.hour
    }

def get_weather_context(config: HeartbeatConfig) -> Dict[str, Any]:
    if not config.weather_api_key:
        return {"available": False}
    
    try:
        import requests
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": config.latitude,
            "lon": config.longitude,
            "appid": config.weather_api_key,
            "units": "imperial"
        }
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        weather = data.get("weather", [{}])[0]
        main = data.get("main", {})
        
        return {
            "available": True,
            "condition": sanitize_text(weather.get("main", "Unknown")),
            "description": sanitize_text(weather.get("description", "Unknown")),
            "temp_f": round(main.get("temp", 0)),
            "summary": f"{weather.get('description', '').title()}, {round(main.get('temp', 0))}°F"
        }
    except Exception as e:
        logger.warning(f"Weather fetch failed: {type(e).__name__}")
        return {"available": False}

def get_last_interaction_time(memory: MemorySystem) -> Tuple[float, Optional[Dict]]:
    try:
        most_recent_ts = 0
        most_recent_turn = None
        
        if FLIGHT_DIR.exists():
            for f in FLIGHT_DIR.glob("flight_recorder_*.jsonl"):
                if not re.match(r'^flight_recorder_[\w-]+\.jsonl$', f.name):
                    continue
                    
                turns = read_rec(f.stem.replace("flight_recorder_", ""))
                if turns:
                    last_turn = turns[-1]
                    ts = last_turn.get('ts', 0)
                    if ts > most_recent_ts:
                        most_recent_ts = ts
                        most_recent_turn = last_turn
        
        if most_recent_ts == 0:
            return float('inf'), None
        
        return time.time() - most_recent_ts, most_recent_turn
    except Exception as e:
        logger.error(f"Error reading interaction time: {e}")
        return float('inf'), None

# =============================================================================
# LLM DECISION GATE
# =============================================================================

def validate_llm_response(data: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "should_speak": False,
        "confidence": 0.0,
        "reason": "Validation failed",
        "message": "",
        "tone": "neutral"
    }
    
    if not isinstance(data, dict):
        logger.warning("LLM returned non-dict")
        return defaults
        
    safe_data = defaults.copy()
    
    if isinstance(data.get("should_speak"), bool):
        safe_data["should_speak"] = data["should_speak"]
        
    try:
        conf = float(data.get("confidence", 0))
        safe_data["confidence"] = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        pass
        
    safe_data["reason"] = sanitize_text(str(data.get("reason", "")), limit=CONFIG.max_message_length)
    safe_data["message"] = sanitize_text(str(data.get("message", "")), limit=CONFIG.max_message_length)
    safe_data["tone"] = sanitize_text(str(data.get("tone", "")), limit=100)
    
    return safe_data

async def make_llm_decision_async(
    context: Dict[str, Any],
    memories: Optional[str],
    mood: MoodState,
    config: HeartbeatConfig
) -> Dict[str, Any]:
    
    safe_memories = sanitize_text(memories or "", limit=config.max_memory_length)
    
    prompt = f"""You are the decision engine for an AI companion named "{config.companion_name}".
Decide if you should proactively reach out to Stephen.

CONTEXT:
Time: {context['time']['day_of_week']} {context['time']['time_of_day']}
Weather: {context['weather'].get('summary', 'N/A')}
Hours since last interaction: {context['last_interaction']['hours_ago']:.1f}

STATE:
Mood: {mood.current_mood} (Energy: {mood.energy_level:.1f})

MEMORIES:
{safe_memories}

INSTRUCTIONS:
1. ONLY speak if there is a compelling reason (weather alert, specific memory topic, time relevance).
2. Do NOT be spammy.
3. Return JSON.

JSON FORMAT:
{{
    "should_speak": boolean,
    "confidence": float (0.0-1.0),
    "reason": "string",
    "message": "string"
}}
"""

    # FIX: Sync functions for to_thread
    def _call_openai():
        from openai import OpenAI
        # Note: Injecting API Key here if needed, though client typically grabs from env
        client = OpenAI(timeout=config.request_timeout)
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=250,
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)

    def _call_gemini():
        import google.generativeai as genai
        # Note: Ensure genai is configured in main/env or here
        # We assume global config or env var handles it
        model = genai.GenerativeModel('gemini-2.0-flash-exp', generation_config={"response_mime_type": "application/json"})
        result = model.generate_content(prompt)
        return json.loads(result.text)

    try:
        raw_json = {}
        if config.llm_provider == "openai":
            raw_json = await asyncio.to_thread(_call_openai)
            
        elif config.llm_provider == "gemini":
            # Configure Gemini if not already
            if not os.getenv("GEMINI_API_KEY"):
                 return {"should_speak": False, "reason": "Gemini Key Missing"}
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            raw_json = await asyncio.wait_for(
                asyncio.to_thread(_call_gemini), 
                timeout=config.request_timeout
            )
            
        else:
            return {"should_speak": False, "reason": "Config error"}

        return validate_llm_response(raw_json)

    except asyncio.TimeoutError:
        logger.error("LLM request timed out")
        return {"should_speak": False, "reason": "Timeout"}
    except Exception as e:
        logger.error(f"LLM decision failed: {type(e).__name__}")
        return {"should_speak": False, "reason": "LLM Error"}

# =============================================================================
# DELIVERY
# =============================================================================

def send_toast_notification(title: str, message: str, timeout: int = 10) -> bool:
    try:
        from plyer import notification
        clean_msg = sanitize_text(message, limit=500)
        
        notification.notify(
            title=sanitize_text(title, limit=100),
            message=clean_msg,
            app_name="AI Companion",
            timeout=timeout
        )
        return True
    except Exception as e:
        logger.error(f"Toast failed: {e}")
        return False

# =============================================================================
# ENGINE
# =============================================================================

class HeartbeatEngine:
    def __init__(self, config: HeartbeatConfig):
        self.config = config
        self.config.validate()
        self.state_file = Path(config.state_file)
        
        # Initialize MemorySystem from imported module
        try:
            self.memory = MemorySystem()
        except Exception as e:
            logger.error(f"Failed to init MemorySystem: {e}")
            sys.exit(1)
            
        self.mood = MoodManager(config.mood_file)
        self.state = self._load_state()
        
        logger.info("🫀 Heartbeat Engine V1.2.1 Initialized")
        logger.info(f"   Provider: {config.llm_provider}")
        logger.info(f"   Weather Key: {mask_key(config.weather_api_key)}")

    def _load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding='utf-8'))
            except Exception:
                logger.warning("State file corrupt, starting fresh.")
        return {
            "messages_sent_today": 0,
            "last_message_ts": 0,
            "last_check_day": datetime.now().day
        }

    def _save_state(self) -> None:
        try:
            atomic_write(self.state_file, self.state)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    def _check_constraints(self) -> Tuple[bool, str]:
        today = datetime.now().day
        if today != self.state.get("last_check_day"):
            self.state["messages_sent_today"] = 0
            self.state["last_check_day"] = today
            self._save_state()

        hour = datetime.now().hour
        s, e = self.config.quiet_hours_start, self.config.quiet_hours_end
        is_quiet = (hour >= s or hour < e) if s > e else (s <= hour < e)
        if is_quiet: return False, "Quiet hours"

        if self.state["messages_sent_today"] >= self.config.max_daily_messages:
            return False, "Daily limit"
            
        if time.time() - self.state["last_message_ts"] < self.config.min_gap_seconds:
            return False, "Minimum gap"
            
        return True, "OK"

    async def cycle(self) -> None:
        can_send, reason = self._check_constraints()
        if not can_send:
            logger.debug(f"Blocked: {reason}")
            return

        # Gather Context
        context = {
            "time": get_time_context(),
            "weather": get_weather_context(self.config),
            "mood": asdict(self.mood.update()),
        }
        
        ago, turn = get_last_interaction_time(self.memory)
        context["last_interaction"] = {"hours_ago": ago / 3600 if ago != float('inf') else 999}

        # Retrieval
        query = f"It is {context['time']['day_of_week']}. Weather is {context['weather'].get('condition','unknown')}."
        memories = None
        try:
            memories = await asyncio.to_thread(
                self.memory.proactive_retrieval, 
                sanitize_text(query, limit=500), 
                ""
            )
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")

        # Async Decision with Timeout
        decision = await make_