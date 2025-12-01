"""
The Soul: Emotional State Machine v2.0
Persists personality traits that the LLM context window forgets.
"""
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# Setup logging
logger = logging.getLogger("Soul")

@dataclass
class SoulState:
    """The persistent 'ghost' in the machine."""
    mood: str = "neutral"
    energy: float = 0.8          # 0.0 (Exhausted) to 1.0 (Wired)
    intimacy: float = 0.5        # 0.0 (Stranger) to 1.0 (Soulmate)
    last_interaction: float = 0.0
    total_turns: int = 0

class EmotionalState:
    def __init__(self, state_file="soul_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load_state()

    def _load_state(self) -> SoulState:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding='utf-8'))
                return SoulState(**data)
            except Exception as e:
                logger.error(f"Corrupt soul file: {e}")
        return SoulState(last_interaction=time.time())

    def save(self):
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.state), f, indent=2)

    def update_circadian_rhythm(self):
        """Adjusts Energy based on time of day (Biomimicry)."""
        hour = datetime.now().hour
        
        # Morning ramp-up (6 AM - 11 AM)
        if 6 <= hour < 11:
            self.state.energy = min(1.0, self.state.energy + 0.1)
        # Evening wind-down (9 PM - 5 AM)
        elif 21 <= hour or hour < 5:
            self.state.energy = max(0.2, self.state.energy - 0.1)
            
        # Intimacy Decay: Absence makes the heart grow... distant?
        hours_since = (time.time() - self.state.last_interaction) / 3600
        if hours_since > 24:
            decay = 0.01 * (hours_since / 24)
            self.state.intimacy = max(0.3, self.state.intimacy - decay)

    def register_interaction(self, sentiment="positive"):
        """Update state based on interaction."""
        self.state.last_interaction = time.time()
        self.state.total_turns += 1
        
        if sentiment == "positive":
            self.state.intimacy = min(1.0, self.state.intimacy + 0.01)
            self.state.mood = "warm"
        elif sentiment == "negative":
            self.state.intimacy = max(0.0, self.state.intimacy - 0.05)
            self.state.mood = "concerned"
            
        self.save()

    def get_system_prompt(self) -> str:
        """Generates the hidden context for the LLM."""
        energy_desc = "High/Energetic" if self.state.energy > 0.7 else "Low/Sleepy"
        intimacy_desc = "Close Partner" if self.state.intimacy > 0.8 else "Friend"
        
        return (
            f"[INTERNAL SOUL STATE]\n"
            f"MOOD: {self.state.mood}\n"
            f"ENERGY: {self.state.energy:.2f} ({energy_desc})\n"
            f"INTIMACY: {self.state.intimacy:.2f} ({intimacy_desc})\n"
            f"INSTRUCTION: Act consistently with this state. "
            f"If energy is low, be concise. If intimacy is high, be warmer/softer."
        )