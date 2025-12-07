"""
Heartbeat Service V2.0 - Proactive AI Companion
Transforms the AI from reactive assistant to proactive companion.

Features:
- Periodic decision cycle (should I reach out?)
- Context-aware messaging (time, weather, memories)
- Respects quiet hours and daily limits
- Integrates with Soul emotional state
- Weather-aware greetings
"""
import os
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

logger = logging.getLogger("Heartbeat")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HeartbeatConfig:
    """Heartbeat configuration with sensible defaults."""
    # Timing
    interval_seconds: int = 300        # Check every 5 minutes
    min_gap_seconds: int = 7200        # Minimum 2 hours between messages
    quiet_hours_start: int = 22        # 10 PM
    quiet_hours_end: int = 7           # 7 AM
    
    # Limits
    max_daily_messages: int = 3
    max_memory_length: int = 2000
    max_message_length: int = 500
    
    # Location (for weather)
    location: str = "Arroyo Grande, CA"
    latitude: float = 35.1186
    longitude: float = -120.5910
    
    # LLM
    decision_threshold: float = 0.7
    
    # Personality
    companion_name: str = "Her"
    
    @classmethod
    def from_env(cls) -> 'HeartbeatConfig':
        """Load config from environment variables."""
        return cls(
            interval_seconds=int(os.getenv("HEARTBEAT_INTERVAL", "300")),
            min_gap_seconds=int(os.getenv("HEARTBEAT_MIN_GAP", "7200")),
            quiet_hours_start=int(os.getenv("HEARTBEAT_QUIET_START", "22")),
            quiet_hours_end=int(os.getenv("HEARTBEAT_QUIET_END", "7")),
            max_daily_messages=int(os.getenv("HEARTBEAT_MAX_DAILY", "3")),
            location=os.getenv("HEARTBEAT_LOCATION", "Arroyo Grande, CA"),
            latitude=float(os.getenv("HEARTBEAT_LAT", "35.1186")),
            longitude=float(os.getenv("HEARTBEAT_LON", "-120.5910")),
        )


# =============================================================================
# HEARTBEAT STATE
# =============================================================================

@dataclass
class HeartbeatState:
    """Persistent state for heartbeat decisions."""
    messages_sent_today: int = 0
    last_message_ts: float = 0
    last_check_day: int = 0
    last_decision: str = ""
    pending_message: Optional[str] = None
    
    def reset_daily(self):
        """Reset daily counters if new day."""
        today = datetime.now().day
        if today != self.last_check_day:
            self.messages_sent_today = 0
            self.last_check_day = today


class HeartbeatStateManager:
    """Manages heartbeat state persistence."""
    
    def __init__(self, state_file: str = "heartbeat_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load()
    
    def _load(self) -> HeartbeatState:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding='utf-8'))
                return HeartbeatState(**data)
            except Exception as e:
                logger.warning(f"Heartbeat state corrupt, resetting: {e}")
        return HeartbeatState(last_check_day=datetime.now().day)
    
    def save(self):
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save heartbeat state: {e}")
    
    def record_message_sent(self, message: str):
        """Record that a message was sent."""
        self.state.messages_sent_today += 1
        self.state.last_message_ts = time.time()
        self.state.pending_message = None
        self.save()


# =============================================================================
# CONTEXT GATHERERS
# =============================================================================

def get_time_context() -> Dict[str, Any]:
    """Get current time context."""
    now = datetime.now()
    hour = now.hour
    
    # Determine greeting type
    if 5 <= hour < 12:
        greeting_type = "morning"
    elif 12 <= hour < 17:
        greeting_type = "afternoon"
    elif 17 <= hour < 21:
        greeting_type = "evening"
    else:
        greeting_type = "night"
    
    return {
        "datetime": now.isoformat(),
        "day_of_week": now.strftime("%A"),
        "time_of_day": now.strftime("%I:%M %p"),
        "hour": hour,
        "greeting_type": greeting_type,
        "is_weekend": now.weekday() >= 5
    }


async def get_weather_context(config: HeartbeatConfig) -> Dict[str, Any]:
    """Fetch weather from OpenWeatherMap."""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return {"available": False, "reason": "No API key"}
    
    try:
        import requests
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": config.latitude,
            "lon": config.longitude,
            "appid": api_key,
            "units": "imperial"
        }
        
        response = await asyncio.to_thread(
            lambda: requests.get(url, params=params, timeout=5)
        )
        response.raise_for_status()
        data = response.json()
        
        weather = data.get("weather", [{}])[0]
        main = data.get("main", {})
        
        return {
            "available": True,
            "condition": weather.get("main", "Unknown"),
            "description": weather.get("description", "Unknown"),
            "temp_f": round(main.get("temp", 0)),
            "feels_like": round(main.get("feels_like", 0)),
            "humidity": main.get("humidity", 0),
            "summary": f"{weather.get('description', '').title()}, {round(main.get('temp', 0))}°F"
        }
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        return {"available": False, "reason": str(e)}


# =============================================================================
# DECISION ENGINE
# =============================================================================

async def make_heartbeat_decision(
    context: Dict[str, Any],
    memories: Optional[str],
    soul_state: Dict[str, Any],
    config: HeartbeatConfig
) -> Dict[str, Any]:
    """
    Use Gemini to decide if we should proactively reach out.
    Returns: {should_speak, confidence, reason, message, tone}
    """
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"should_speak": False, "reason": "No Gemini API key"}
        genai.configure(api_key=api_key)
    except ImportError:
        return {"should_speak": False, "reason": "Gemini not available"}
    
    # Build decision prompt
    prompt = f"""You are the decision engine for an AI companion named "{config.companion_name}".
Decide if you should proactively reach out to Stephen right now.

CURRENT CONTEXT:
- Time: {context['time']['day_of_week']} {context['time']['time_of_day']} ({context['time']['greeting_type']})
- Weekend: {context['time']['is_weekend']}
- Weather: {context.get('weather', {}).get('summary', 'Unknown')}
- Hours since last interaction: {context.get('hours_since_interaction', 'Unknown')}

YOUR EMOTIONAL STATE:
- Mood: {soul_state.get('mood', 'neutral')}
- Energy: {soul_state.get('energy', 0.5):.1f}
- Intimacy with user: {soul_state.get('intimacy', 0.5):.1f}
- Circadian phase: {soul_state.get('circadian_phase', 'day')}

RELEVANT MEMORIES:
{memories or 'No specific memories triggered.'}

DECISION RULES:
1. Only speak if there's a COMPELLING reason (weather event, relevant memory, time-based check-in)
2. Do NOT be spammy - quality over quantity
3. Consider your energy level - if low, maybe just a brief check-in
4. Consider intimacy level - higher intimacy = warmer, more personal messages
5. Weather events (storms, extreme temps) are good reasons to reach out
6. Weekends are good for casual check-ins if intimacy is high
7. Don't reach out during work hours (9-5) unless something important

Return JSON:
{{
    "should_speak": boolean,
    "confidence": float (0.0-1.0),
    "reason": "brief explanation",
    "message": "the message to send if should_speak is true",
    "tone": "warm|playful|concerned|curious|supportive"
}}
"""

    try:
        model = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config={"response_mime_type": "application/json"}
        )
        result = await asyncio.to_thread(model.generate_content, prompt)
        decision = json.loads(result.text)
        
        # Validate response
        decision.setdefault("should_speak", False)
        decision.setdefault("confidence", 0.0)
        decision.setdefault("reason", "")
        decision.setdefault("message", "")
        decision.setdefault("tone", "neutral")
        
        # Apply confidence threshold
        if decision["confidence"] < config.decision_threshold:
            decision["should_speak"] = False
            decision["reason"] = f"Confidence {decision['confidence']:.2f} below threshold {config.decision_threshold}"
        
        return decision
    except Exception as e:
        logger.error(f"Heartbeat decision failed: {e}")
        return {"should_speak": False, "reason": f"Decision error: {e}"}


# =============================================================================
# HEARTBEAT ENGINE
# =============================================================================

class HeartbeatEngine:
    """
    The Heartbeat: Transforms AI from reactive to proactive.
    
    Runs as a background task, periodically deciding whether to reach out.
    Integrates with Soul for emotional awareness and MemorySystem for context.
    """
    
    def __init__(self, memory_system, soul, config: Optional[HeartbeatConfig] = None):
        self.memory = memory_system
        self.soul = soul
        self.config = config or HeartbeatConfig.from_env()
        self.state_manager = HeartbeatStateManager()
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info("❤️ Heartbeat Engine V2.0 Initialized")
        logger.info(f"   Interval: {self.config.interval_seconds}s")
        logger.info(f"   Quiet hours: {self.config.quiet_hours_start}:00 - {self.config.quiet_hours_end}:00")
    
    def _check_constraints(self) -> Tuple[bool, str]:
        """Check if we're allowed to send a message."""
        self.state_manager.state.reset_daily()
        state = self.state_manager.state
        
        # Check quiet hours
        hour = datetime.now().hour
        start, end = self.config.quiet_hours_start, self.config.quiet_hours_end
        if start > end:  # Wraps around midnight
            is_quiet = hour >= start or hour < end
        else:
            is_quiet = start <= hour < end
        
        if is_quiet:
            return False, "Quiet hours"
        
        # Check daily limit
        if state.messages_sent_today >= self.config.max_daily_messages:
            return False, f"Daily limit ({self.config.max_daily_messages}) reached"
        
        # Check minimum gap
        time_since_last = time.time() - state.last_message_ts
        if time_since_last < self.config.min_gap_seconds:
            remaining = (self.config.min_gap_seconds - time_since_last) / 60
            return False, f"Minimum gap not met ({remaining:.0f}min remaining)"
        
        return True, "OK"
    
    async def _gather_context(self) -> Dict[str, Any]:
        """Gather all context for decision making."""
        context = {
            "time": get_time_context(),
            "weather": await get_weather_context(self.config),
        }
        
        # Get hours since last interaction from flight recorder
        try:
            from flight_recorder_service import FLIGHT_DIR, read_rec
            most_recent_ts = 0
            if FLIGHT_DIR.exists():
                for f in FLIGHT_DIR.glob("flight_recorder_*.jsonl"):
                    turns = read_rec(f.stem.replace("flight_recorder_", ""), "default_user")
                    if turns:
                        ts = turns[-1].get('ts', 0)
                        if ts > most_recent_ts:
                            most_recent_ts = ts
            
            if most_recent_ts > 0:
                context["hours_since_interaction"] = (time.time() - most_recent_ts) / 3600
            else:
                context["hours_since_interaction"] = 999
        except Exception as e:
            logger.warning(f"Could not get interaction time: {e}")
            context["hours_since_interaction"] = 999
        
        return context
    
    async def _get_relevant_memories(self, context: Dict[str, Any]) -> Optional[str]:
        """Retrieve memories relevant to current context."""
        try:
            # Build query from context
            query_parts = [context['time']['day_of_week']]
            if context['weather'].get('available'):
                query_parts.append(context['weather'].get('condition', ''))
            
            query = " ".join(query_parts)
            
            memories = await asyncio.to_thread(
                self.memory.proactive_retrieval,
                "default_user",
                query,
                "global",
                context.get("hours_since_interaction", 0) * 3600
            )
            return memories
        except Exception as e:
            logger.error(f"Memory retrieval for heartbeat failed: {e}")
            return None
    
    async def cycle(self) -> Dict[str, Any]:
        """
        Run one heartbeat cycle.
        Returns the decision dict (useful for testing/monitoring).
        """
        # Check constraints
        can_send, reason = self._check_constraints()
        if not can_send:
            logger.debug(f"Heartbeat blocked: {reason}")
            return {"should_speak": False, "reason": reason}
        
        # Gather context
        context = await self._gather_context()
        
        # Get relevant memories
        memories = await self._get_relevant_memories(context)
        
        # Get soul state
        soul_state = self.soul.get_emotional_context()
        
        # Make decision
        decision = await make_heartbeat_decision(
            context, memories, soul_state, self.config
        )
        
        # Log decision
        if decision.get("should_speak"):
            logger.info(f"💬 Heartbeat wants to speak: {decision.get('reason')}")
            self.state_manager.state.pending_message = decision.get("message")
            self.state_manager.state.last_decision = decision.get("reason")
            self.state_manager.save()
        else:
            logger.debug(f"Heartbeat silent: {decision.get('reason')}")
        
        return decision
    
    async def _loop(self):
        """Main heartbeat loop."""
        logger.info("❤️ Heartbeat loop started")
        while self.running:
            try:
                await self.cycle()
            except Exception as e:
                logger.error(f"Heartbeat cycle error: {e}")
            
            await asyncio.sleep(self.config.interval_seconds)
    
    def start(self):
        """Start the heartbeat background task."""
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("❤️ Heartbeat started")
    
    def stop(self):
        """Stop the heartbeat background task."""
        self.running = False
        if self._task:
            self._task.cancel()
        logger.info("❤️ Heartbeat stopped")
    
    def get_pending_message(self) -> Optional[Dict[str, Any]]:
        """Get any pending message from the heartbeat."""
        state = self.state_manager.state
        if state.pending_message:
            return {
                "message": state.pending_message,
                "reason": state.last_decision,
                "timestamp": time.time()
            }
        return None
    
    def acknowledge_message(self):
        """Acknowledge that a pending message was delivered."""
        self.state_manager.record_message_sent(
            self.state_manager.state.pending_message or ""
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current heartbeat status."""
        state = self.state_manager.state
        can_send, reason = self._check_constraints()
        
        return {
            "running": self.running,
            "can_send": can_send,
            "blocked_reason": reason if not can_send else None,
            "messages_sent_today": state.messages_sent_today,
            "max_daily": self.config.max_daily_messages,
            "last_message_ts": state.last_message_ts,
            "pending_message": state.pending_message is not None,
            "interval_seconds": self.config.interval_seconds,
            "quiet_hours": f"{self.config.quiet_hours_start}:00 - {self.config.quiet_hours_end}:00"
        }

