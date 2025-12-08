"""
The Soul: Emotional State Machine v2.1.0
Persists personality traits that the LLM context window forgets.

V2.1.0 Changes (Phase 1 - Associative Recall Enhancement):
- Added VAD (Valence-Arousal-Dominance) emotional model
- Emotional transitions with triggers
- Enhanced circadian rhythm with seasonal awareness
- Emotional memory tagging support
- Mood history tracking
"""
import json
import time
import logging
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logger = logging.getLogger("Soul")

# =============================================================================
# EMOTIONAL CONSTANTS (Phase 1)
# =============================================================================

# VAD (Valence-Arousal-Dominance) emotion mappings
# Each emotion has a typical VAD signature
EMOTION_VAD_MAP: Dict[str, Tuple[float, float, float]] = {
    # Positive emotions
    "joyful": (0.9, 0.8, 0.7),
    "excited": (0.8, 0.9, 0.6),
    "content": (0.7, 0.3, 0.6),
    "peaceful": (0.6, 0.2, 0.5),
    "warm": (0.7, 0.4, 0.6),
    "loving": (0.9, 0.5, 0.4),
    "playful": (0.8, 0.7, 0.5),
    "curious": (0.6, 0.6, 0.5),
    "hopeful": (0.7, 0.5, 0.5),
    
    # Neutral emotions
    "neutral": (0.5, 0.4, 0.5),
    "thoughtful": (0.5, 0.3, 0.6),
    "focused": (0.5, 0.5, 0.7),
    
    # Negative emotions (still valid states)
    "concerned": (0.3, 0.5, 0.4),
    "anxious": (0.2, 0.8, 0.3),
    "sad": (0.2, 0.2, 0.3),
    "melancholic": (0.3, 0.3, 0.4),
    "tired": (0.4, 0.1, 0.3),
    "frustrated": (0.2, 0.7, 0.5),
    "lonely": (0.2, 0.3, 0.3),
}

# Emotional transition triggers
# Valid mood states
VALID_MOODS = [
    "neutral", "warm", "thoughtful", "concerned", "joyful", 
    "lonely", "tired", "playful", "curious", "excited", "calm"
]

MOOD_TRIGGERS: Dict[str, Dict[str, Any]] = {
    "greeting_morning": {"target": "warm", "valence_boost": 0.1},
    "deep_conversation": {"target": "thoughtful", "intimacy_boost": 0.02},
    "user_shares_problem": {"target": "concerned", "intimacy_boost": 0.03},
    "user_shares_joy": {"target": "joyful", "valence_boost": 0.2},
    "long_absence": {"target": "lonely", "intimacy_decay": 0.05},
    "late_night": {"target": "tired", "energy_decay": 0.15},
    "playful_exchange": {"target": "playful", "valence_boost": 0.1},
    "philosophical_topic": {"target": "curious", "arousal_boost": 0.1},
}

# Circadian rhythm phases
CIRCADIAN_PHASES = {
    "dawn": (5, 7),      # 5-7 AM
    "morning": (7, 12),  # 7 AM - 12 PM
    "afternoon": (12, 17), # 12 - 5 PM
    "evening": (17, 21), # 5 - 9 PM
    "night": (21, 24),   # 9 PM - midnight
    "late_night": (0, 5) # Midnight - 5 AM
}


@dataclass
class VADState:
    """Valence-Arousal-Dominance emotional model."""
    valence: float = 0.5      # -1 (negative) to +1 (positive), stored as 0-1
    arousal: float = 0.4      # 0 (calm) to 1 (excited)
    dominance: float = 0.5    # 0 (submissive) to 1 (dominant)
    
    def to_dict(self) -> Dict[str, float]:
        return {"valence": self.valence, "arousal": self.arousal, "dominance": self.dominance}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'VADState':
        return cls(
            valence=data.get("valence", 0.5),
            arousal=data.get("arousal", 0.4),
            dominance=data.get("dominance", 0.5)
        )
    
    def distance_to(self, other: 'VADState') -> float:
        """Calculate emotional distance (useful for emotional search)."""
        return (
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        ) ** 0.5


@dataclass
class SoulState:
    """The persistent 'ghost' in the machine - enhanced with VAD."""
    mood: str = "neutral"
    energy: float = 0.7          # 0.0 (Exhausted) to 1.0 (Wired)
    intimacy: float = 0.1        # 0.0 (Stranger) to 1.0 (Soulmate) - starts low!
    last_interaction: float = 0.0
    total_turns: int = 0
    
    # V2.1.0: VAD emotional state
    valence: float = 0.5         # Emotional positivity
    arousal: float = 0.4         # Emotional intensity
    dominance: float = 0.5       # Sense of control
    
    # V2.1.0: Tracking
    circadian_phase: str = "day"
    mood_history: List[Dict[str, Any]] = field(default_factory=list)
    consecutive_interactions: int = 0
    
    @property
    def vad(self) -> VADState:
        return VADState(self.valence, self.arousal, self.dominance)
    
    def set_vad(self, vad: VADState):
        self.valence = vad.valence
        self.arousal = vad.arousal
        self.dominance = vad.dominance

class EmotionalState:
    """
    Enhanced Emotional State Machine v2.1.0
    
    Features:
    - VAD (Valence-Arousal-Dominance) emotional model
    - Circadian rhythm with phase awareness
    - Emotional transition triggers
    - Mood history tracking
    - Emotional distance calculations for search
    """
    
    def __init__(self, state_file="soul_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load_state()
        self._update_circadian_phase()

    def _load_state(self) -> SoulState:
        """Load state with improved error handling."""
        if not self.state_file.exists():
            logger.info("No soul state file found, creating new state")
            return SoulState(last_interaction=time.time())
        
        try:
            data = json.loads(self.state_file.read_text(encoding='utf-8'))
            # Handle missing new fields gracefully
            if 'mood_history' not in data:
                data['mood_history'] = []
            return SoulState(**{k: v for k, v in data.items() if k in SoulState.__dataclass_fields__})
        except json.JSONDecodeError as e:
            logger.error(f"Corrupt soul file (invalid JSON): {e}")
            backup_path = self.state_file.with_suffix('.json.corrupt')
            try:
                self.state_file.rename(backup_path)
                logger.warning(f"Backed up corrupt file to {backup_path}")
            except Exception as backup_err:
                logger.error(f"Could not backup corrupt file: {backup_err}")
            return SoulState(last_interaction=time.time())
        except (PermissionError, IOError) as e:
            logger.error(f"Cannot read soul file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading soul state: {e}")
            return SoulState(last_interaction=time.time())

    def save(self):
        """Save state to disk."""
        try:
            # Keep only last 50 mood history entries
            if len(self.state.mood_history) > 50:
                self.state.mood_history = self.state.mood_history[-50:]
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save soul state: {e}")

    def _update_circadian_phase(self) -> str:
        """Determine current circadian phase based on time."""
        hour = datetime.now().hour
        
        for phase, (start, end) in CIRCADIAN_PHASES.items():
            if start <= hour < end:
                self.state.circadian_phase = phase
                return phase
        
        # Handle wrap-around for late_night
        if hour >= 0 and hour < 5:
            self.state.circadian_phase = "late_night"
        
        return self.state.circadian_phase

    def update_circadian_rhythm(self):
        """Adjusts Energy and VAD based on time of day (Biomimicry)."""
        hour = datetime.now().hour
        phase = self._update_circadian_phase()
        
        # Energy adjustments by phase
        energy_adjustments = {
            "dawn": 0.15,
            "morning": 0.1,
            "afternoon": 0.0,
            "evening": -0.05,
            "night": -0.1,
            "late_night": -0.15
        }
        
        adjustment = energy_adjustments.get(phase, 0)
        self.state.energy = max(0.1, min(1.0, self.state.energy + adjustment))
        
        # Arousal tends to decrease late at night
        if phase in ["night", "late_night"]:
            self.state.arousal = max(0.1, self.state.arousal - 0.05)
        elif phase in ["dawn", "morning"]:
            self.state.arousal = min(0.8, self.state.arousal + 0.05)
            
        # Intimacy Decay: Absence makes the heart grow... distant?
        hours_since = (time.time() - self.state.last_interaction) / 3600
        if hours_since > 24:
            decay = 0.01 * (hours_since / 24)
            self.state.intimacy = max(0.3, self.state.intimacy - decay)
            
            # Long absence triggers loneliness
            if hours_since > 48:
                self._trigger_transition("long_absence")

    def _trigger_transition(self, trigger_name: str) -> bool:
        """Apply an emotional transition trigger."""
        if trigger_name not in MOOD_TRIGGERS:
            return False
        
        trigger = MOOD_TRIGGERS[trigger_name]
        
        # Update mood
        if "target" in trigger:
            self._set_mood(trigger["target"])
        
        # Apply VAD adjustments
        if "valence_boost" in trigger:
            self.state.valence = min(1.0, self.state.valence + trigger["valence_boost"])
        if "arousal_boost" in trigger:
            self.state.arousal = min(1.0, self.state.arousal + trigger["arousal_boost"])
        if "intimacy_boost" in trigger:
            self.state.intimacy = min(1.0, self.state.intimacy + trigger["intimacy_boost"])
        if "energy_decay" in trigger:
            self.state.energy = max(0.1, self.state.energy - trigger["energy_decay"])
        if "intimacy_decay" in trigger:
            self.state.intimacy = max(0.3, self.state.intimacy - trigger["intimacy_decay"])
        
        logger.debug(f"Emotional trigger '{trigger_name}' applied -> mood: {self.state.mood}")
        return True
    
    def _set_mood(self, mood: str) -> None:
        """Set mood and update VAD to match."""
        if mood in EMOTION_VAD_MAP:
            v, a, d = EMOTION_VAD_MAP[mood]
            # Blend toward target VAD (don't snap instantly)
            self.state.valence = 0.7 * self.state.valence + 0.3 * v
            self.state.arousal = 0.7 * self.state.arousal + 0.3 * a
            self.state.dominance = 0.7 * self.state.dominance + 0.3 * d
        
        old_mood = self.state.mood
        self.state.mood = mood
        
        # Track mood history
        self.state.mood_history.append({
            "ts": time.time(),
            "from": old_mood,
            "to": mood,
            "vad": self.state.vad.to_dict()
        })

    def register_interaction(self, sentiment: str = "positive", trigger: Optional[str] = None):
        """
        Update state based on interaction.
        
        Args:
            sentiment: "positive", "negative", or "neutral"
            trigger: Optional specific trigger name from MOOD_TRIGGERS
        """
        self.state.last_interaction = time.time()
        self.state.total_turns += 1
        self.state.consecutive_interactions += 1
        
        # Apply specific trigger if provided
        if trigger:
            self._trigger_transition(trigger)
        else:
            # Default sentiment-based update
            if sentiment == "positive":
                self.state.intimacy = min(1.0, self.state.intimacy + 0.01)
                self.state.valence = min(1.0, self.state.valence + 0.05)
                self._set_mood("warm")
            elif sentiment == "negative":
                self.state.intimacy = max(0.0, self.state.intimacy - 0.02)
                self.state.valence = max(0.0, self.state.valence - 0.1)
                self._set_mood("concerned")
            else:  # neutral
                self._set_mood("thoughtful")
        
        # Consecutive interaction bonus
        if self.state.consecutive_interactions >= 5:
            self.state.intimacy = min(1.0, self.state.intimacy + 0.02)
            
        self.save()
    
    def register_absence(self):
        """Called when a session ends or significant time passes."""
        self.state.consecutive_interactions = 0
        self.save()

    def get_current_vad(self) -> VADState:
        """Get current VAD state for emotional search/tagging."""
        return self.state.vad
    
    def get_emotion_for_memory(self) -> Dict[str, Any]:
        """
        Get emotional metadata for tagging a new memory.
        Used by memory_service when storing memories.
        """
        # Find closest named emotion to current VAD
        current = self.state.vad
        closest_emotion = "neutral"
        min_distance = float('inf')
        
        for emotion, (v, a, d) in EMOTION_VAD_MAP.items():
            target = VADState(v, a, d)
            dist = current.distance_to(target)
            if dist < min_distance:
                min_distance = dist
                closest_emotion = emotion
        
        return {
            "primary": closest_emotion,
            "secondary": self._get_secondary_emotions(),
            "valence": round(self.state.valence, 3),
            "arousal": round(self.state.arousal, 3),
            "dominance": round(self.state.dominance, 3)
        }
    
    def _get_secondary_emotions(self) -> List[str]:
        """Get 2-3 secondary emotions near current VAD."""
        current = self.state.vad
        distances = []
        
        for emotion, (v, a, d) in EMOTION_VAD_MAP.items():
            target = VADState(v, a, d)
            dist = current.distance_to(target)
            distances.append((emotion, dist))
        
        # Sort by distance and take 2nd-4th closest
        distances.sort(key=lambda x: x[1])
        return [e for e, _ in distances[1:4]]

    def get_system_prompt(self) -> str:
        """Generates the hidden context for the LLM."""
        energy_desc = "High/Energetic" if self.state.energy > 0.7 else ("Moderate" if self.state.energy > 0.4 else "Moderate")
        intimacy_desc = "Close Colleague" if self.state.intimacy > 0.85 else ("Familiar" if self.state.intimacy > 0.5 else "Professional")
        valence_desc = "Positive" if self.state.valence > 0.6 else ("Negative" if self.state.valence < 0.4 else "Neutral")
        arousal_desc = "Engaged" if self.state.arousal > 0.7 else ("Calm" if self.state.arousal < 0.3 else "Moderate")
        
        return (
            f"[INTERNAL STATE]\n"
            f"FAMILIARITY: {intimacy_desc}\n"
            f"\nRULES: Answer directly then STOP. Never end with questions or offers to help more."
        )
    
    def get_emotional_context(self) -> Dict[str, Any]:
        """Get full emotional context for API responses."""
        return {
            "mood": self.state.mood,
            "energy": round(self.state.energy, 2),
            "intimacy": round(self.state.intimacy, 2),
            "vad": self.state.vad.to_dict(),
            "circadian_phase": self.state.circadian_phase,
            "consecutive_interactions": self.state.consecutive_interactions,
            "emotions": self.get_emotion_for_memory()
        }


# =============================================================================
# EMOTIONAL ANALYSIS FUNCTIONS (for memory tagging)
# =============================================================================

def analyze_text_emotion(text: str) -> Optional[Dict[str, Any]]:
    """
    Use Gemini to analyze emotional content of text.
    Returns VAD scores and emotion labels for memory tagging.
    """
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
    except ImportError:
        return None
    
    prompt = f"""Analyze the emotional content of this text and return JSON:

TEXT: "{text}"

Return JSON with:
- primary: main emotion (from: joyful, excited, content, peaceful, warm, loving, playful, curious, hopeful, neutral, thoughtful, focused, concerned, anxious, sad, melancholic, tired, frustrated, lonely)
- secondary: list of 1-2 secondary emotions
- valence: -1 to +1 (negative to positive)
- arousal: 0 to 1 (calm to excited)
- dominance: 0 to 1 (submissive to dominant)

Return ONLY the JSON object."""

    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash-lite',
            generation_config={"response_mime_type": "application/json"}
        )
        result = model.generate_content(prompt)
        data = json.loads(result.text)
        
        # Normalize valence from -1..1 to 0..1 for storage
        if 'valence' in data:
            data['valence'] = (data['valence'] + 1) / 2
        
        return data
    except Exception as e:
        logger.warning(f"Emotion analysis failed: {e}")
        return None


def calculate_emotional_distance(emotion1: Dict[str, float], emotion2: Dict[str, float]) -> float:
    """
    Calculate distance between two emotional states.
    Useful for emotional search - finding memories with similar emotional tone.
    """
    v1 = VADState(
        emotion1.get('valence', 0.5),
        emotion1.get('arousal', 0.4),
        emotion1.get('dominance', 0.5)
    )
    v2 = VADState(
        emotion2.get('valence', 0.5),
        emotion2.get('arousal', 0.4),
        emotion2.get('dominance', 0.5)
    )
    return v1.distance_to(v2)
