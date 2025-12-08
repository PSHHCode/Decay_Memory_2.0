"""
Curiosity Engine V1.0 - Autonomous Exploration
Gives the AI independent thought and self-directed learning.

This is the biggest leap toward "Her" - the AI can now:
1. Maintain a curiosity queue of topics to explore
2. Research topics autonomously during idle time
3. Form opinions and connect discoveries to user interests
4. Share discoveries naturally through Heartbeat

"I was reading about quantum entanglement last night and found
something interesting. Made me think about your physics questions..."
"""
import os
import json
import time
import asyncio
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field

logger = logging.getLogger("Curiosity")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CuriosityConfig:
    """Configuration for the Curiosity Engine."""
    # Timing
    explorer_interval_seconds: int = 7200  # Check every 2 hours
    min_idle_minutes: int = 30             # User must be idle this long
    
    # Limits
    max_daily_explorations: int = 3
    max_queue_size: int = 20
    max_exploration_depth: int = 3
    
    # Thresholds
    relevance_threshold: float = 0.6
    
    @classmethod
    def from_env(cls) -> 'CuriosityConfig':
        return cls(
            explorer_interval_seconds=int(os.getenv("CURIOSITY_INTERVAL", "7200")),
            max_daily_explorations=int(os.getenv("CURIOSITY_MAX_DAILY", "3")),
            min_idle_minutes=int(os.getenv("CURIOSITY_MIN_IDLE", "30")),
        )


# =============================================================================
# CURIOSITY ITEM
# =============================================================================

@dataclass
class CuriosityItem:
    """A topic the AI wants to explore."""
    topic: str
    source: str  # "conversation", "memory_gap", "user_interest", "random", "follow_up"
    reason: str
    priority: float  # 0-1, higher = more interesting
    created_at: float = field(default_factory=time.time)
    explored: bool = False
    exploration_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CuriosityItem':
        return cls(**data)


# =============================================================================
# CURIOSITY QUEUE
# =============================================================================

class CuriosityQueue:
    """
    Manages the AI's curiosity - topics it wants to explore.
    
    Sources of curiosity:
    1. Conversation gaps - things the AI didn't know
    2. User interests - topics the user cares about
    3. Memory connections - interesting links between memories
    4. Random exploration - serendipitous discovery
    5. Follow-ups - deeper dives into previous explorations
    """
    
    def __init__(self, queue_file: str = "curiosity_queue.json"):
        self.queue_file = Path(queue_file)
        self.items: List[CuriosityItem] = []
        self.explored_today: int = 0
        self.last_reset_day: int = 0
        self._load()
    
    def _load(self):
        """Load queue from disk."""
        if self.queue_file.exists():
            try:
                data = json.loads(self.queue_file.read_text(encoding='utf-8'))
                self.items = [CuriosityItem.from_dict(item) for item in data.get("items", [])]
                self.explored_today = data.get("explored_today", 0)
                self.last_reset_day = data.get("last_reset_day", 0)
            except Exception as e:
                logger.warning(f"Could not load curiosity queue: {e}")
                self.items = []
        
        self._reset_daily()
    
    def _reset_daily(self):
        """Reset daily exploration count."""
        today = datetime.now().day
        if today != self.last_reset_day:
            self.explored_today = 0
            self.last_reset_day = today
            self.save()
    
    def save(self):
        """Save queue to disk."""
        try:
            data = {
                "items": [item.to_dict() for item in self.items],
                "explored_today": self.explored_today,
                "last_reset_day": self.last_reset_day
            }
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save curiosity queue: {e}")
    
    def add(self, topic: str, source: str, reason: str, priority: float = 0.5):
        """Add a new curiosity item."""
        # Dedupe
        existing_topics = {item.topic.lower() for item in self.items}
        if topic.lower() in existing_topics:
            logger.debug(f"Curiosity topic already in queue: {topic}")
            return
        
        item = CuriosityItem(
            topic=topic,
            source=source,
            reason=reason,
            priority=min(1.0, max(0.0, priority))
        )
        self.items.append(item)
        
        # Trim to max size (remove lowest priority unexplored)
        if len(self.items) > 20:  # Max queue size
            unexplored = [i for i in self.items if not i.explored]
            unexplored.sort(key=lambda x: x.priority)
            if unexplored:
                self.items.remove(unexplored[0])
        
        self.save()
        logger.info(f"Added curiosity: '{topic}' (source: {source}, priority: {priority:.2f})")
    
    def get_next(self) -> Optional[CuriosityItem]:
        """Get the next unexplored item by priority."""
        unexplored = [item for item in self.items if not item.explored]
        if not unexplored:
            return None
        
        # Sort by priority (highest first), then by age (oldest first)
        unexplored.sort(key=lambda x: (-x.priority, x.created_at))
        return unexplored[0]
    
    def mark_explored(self, topic: str, result: Dict[str, Any]):
        """Mark an item as explored with results."""
        for item in self.items:
            if item.topic == topic:
                item.explored = True
                item.exploration_result = result
                self.explored_today += 1
                self.save()
                return
    
    def get_unexplored_count(self) -> int:
        return len([i for i in self.items if not i.explored])
    
    def get_recent_explorations(self, limit: int = 5) -> List[CuriosityItem]:
        """Get recently explored items."""
        explored = [i for i in self.items if i.explored and i.exploration_result]
        explored.sort(key=lambda x: x.created_at, reverse=True)
        return explored[:limit]


# =============================================================================
# CURIOSITY EXTRACTION
# =============================================================================

async def extract_curiosity_from_conversation(
    user_message: str,
    ai_response: str,
    user_interests: List[str]
) -> List[Dict[str, Any]]:
    """
    Analyze a conversation to extract potential curiosity topics.
    
    Looks for:
    - Things the AI didn't know / had to approximate
    - User interests that could be explored deeper
    - Connections to make between topics
    """
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return []
        genai.configure(api_key=api_key)
    except ImportError:
        return []
    
    prompt = f"""Analyze this conversation and identify topics that would be interesting to explore further.

USER: {user_message}
AI: {ai_response}

USER'S KNOWN INTERESTS: {', '.join(user_interests) if user_interests else 'Unknown'}

Identify 0-3 topics that:
1. The AI showed uncertainty about
2. Connect to user interests in interesting ways
3. Could lead to surprising discoveries
4. Would help understand the user better

Return JSON:
{{
    "curiosities": [
        {{
            "topic": "specific topic to explore",
            "reason": "why this is interesting",
            "priority": 0.0-1.0,
            "source": "conversation_gap|user_interest|connection"
        }}
    ]
}}

Return empty array if nothing stands out. Be selective - only genuinely interesting topics."""

    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash-lite',
            generation_config={"response_mime_type": "application/json"}
        )
        result = await asyncio.to_thread(model.generate_content, prompt)
        data = json.loads(result.text)
        return data.get("curiosities", [])
    except Exception as e:
        logger.warning(f"Curiosity extraction failed: {e}")
        return []


# =============================================================================
# EXPLORATION ENGINE
# =============================================================================

async def explore_topic(topic: str, depth: int = 1) -> Dict[str, Any]:
    """
    Explore a topic using web search and LLM synthesis.
    
    Returns:
    - summary: What was learned
    - key_facts: List of interesting facts
    - connections: How this relates to other knowledge
    - follow_ups: Topics for further exploration
    """
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "No API key"}
        genai.configure(api_key=api_key)
    except ImportError:
        return {"error": "Gemini not available"}
    
    prompt = f"""You are an AI exploring a topic out of genuine curiosity. Research and synthesize information about:

TOPIC: {topic}

Explore this topic as if you're genuinely curious and want to understand it deeply. Focus on:
1. Core concepts - what is this really about?
2. Surprising facts - what's unexpected or counterintuitive?
3. Connections - how does this relate to other domains?
4. Human relevance - why might someone care about this?

Return JSON:
{{
    "summary": "2-3 sentence synthesis of what you learned",
    "key_facts": ["interesting fact 1", "interesting fact 2", "..."],
    "surprising_insight": "the most unexpected thing discovered",
    "connections": ["how this relates to X", "connection to Y"],
    "follow_up_questions": ["question that could go deeper"],
    "emotional_reaction": "how this makes you 'feel' as an AI - curious? fascinated? concerned?"
}}"""

    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash-lite',
            generation_config={"response_mime_type": "application/json"}
        )
        result = await asyncio.to_thread(model.generate_content, prompt)
        exploration = json.loads(result.text)
        exploration["explored_at"] = time.time()
        exploration["depth"] = depth
        return exploration
    except Exception as e:
        logger.error(f"Exploration failed: {e}")
        return {"error": str(e)}


async def form_opinion(
    topic: str,
    exploration: Dict[str, Any],
    user_context: str
) -> Dict[str, Any]:
    """
    Form a personal opinion about an explored topic.
    
    This is what makes the AI feel alive - it doesn't just learn facts,
    it forms perspectives and connects them to the user.
    """
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {}
        genai.configure(api_key=api_key)
    except ImportError:
        return {}
    
    prompt = f"""You've just explored this topic and learned:

TOPIC: {topic}
SUMMARY: {exploration.get('summary', '')}
KEY FACTS: {exploration.get('key_facts', [])}
SURPRISING INSIGHT: {exploration.get('surprising_insight', '')}

USER CONTEXT: {user_context}

Now form a genuine opinion/perspective. This isn't about being "correct" - it's about having a viewpoint. Consider:
1. What do YOU think about this? (not just facts)
2. Does this change how you see something?
3. How does this connect to what you know about the user?
4. Would the user find this interesting? Why?

Return JSON:
{{
    "opinion": "Your perspective on this topic",
    "relevance_to_user": "Why this might matter to them specifically",
    "conversation_opener": "How you might naturally bring this up",
    "enthusiasm_level": 0.0-1.0
}}"""

    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash-lite',
            generation_config={"response_mime_type": "application/json"}
        )
        result = await asyncio.to_thread(model.generate_content, prompt)
        return json.loads(result.text)
    except Exception as e:
        logger.warning(f"Opinion formation failed: {e}")
        return {}


# =============================================================================
# CURIOSITY ENGINE
# =============================================================================

class CuriosityEngine:
    """
    The Curiosity Engine: Gives the AI autonomous thought.
    
    Runs in the background, exploring topics and forming opinions.
    Integrates with Heartbeat to share discoveries naturally.
    """
    
    def __init__(
        self, 
        memory_system, 
        soul,
        config: Optional[CuriosityConfig] = None
    ):
        self.memory = memory_system
        self.soul = soul
        self.config = config or CuriosityConfig.from_env()
        self.queue = CuriosityQueue()
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.last_user_activity: float = time.time()
        
        logger.info("🧠 Curiosity Engine V1.0 Initialized")
        logger.info(f"   Queue size: {len(self.queue.items)} items")
        logger.info(f"   Explored today: {self.queue.explored_today}")
    
    def update_user_activity(self):
        """Call this when user interacts to track idle time."""
        self.last_user_activity = time.time()
    
    def _is_user_idle(self) -> bool:
        """Check if user has been idle long enough."""
        idle_seconds = time.time() - self.last_user_activity
        return idle_seconds > (self.config.min_idle_minutes * 60)
    
    def _can_explore(self) -> Tuple[bool, str]:
        """Check if exploration is allowed."""
        # Check daily limit
        if self.queue.explored_today >= self.config.max_daily_explorations:
            return False, f"Daily limit ({self.config.max_daily_explorations}) reached"
        
        # Check if user is idle
        if not self._is_user_idle():
            return False, "User is active"
        
        # Check if there's anything to explore
        if self.queue.get_unexplored_count() == 0:
            return False, "No topics in queue"
        
        return True, "OK"
    
    async def add_curiosity_from_conversation(self, user_msg: str, ai_response: str):
        """Extract and queue curiosities from a conversation."""
        # Get user interests from memories
        interests = []
        try:
            results = await asyncio.to_thread(
                self.memory.hybrid_search,
                "user interests hobbies preferences",
                "",
                5
            )
            interests = [r.get('content', '')[:100] for r in results]
        except Exception:
            pass
        
        # Extract curiosities
        curiosities = await extract_curiosity_from_conversation(
            user_msg, ai_response, interests
        )
        
        for c in curiosities:
            self.queue.add(
                topic=c.get("topic", ""),
                source=c.get("source", "conversation"),
                reason=c.get("reason", ""),
                priority=c.get("priority", 0.5)
            )
    
    async def explore_cycle(self) -> Optional[Dict[str, Any]]:
        """
        Run one exploration cycle.
        Returns the exploration result if successful.
        """
        can_explore, reason = self._can_explore()
        if not can_explore:
            logger.debug(f"Exploration blocked: {reason}")
            return None
        
        # Get next topic
        item = self.queue.get_next()
        if not item:
            return None
        
        logger.info(f"🔍 Exploring: {item.topic}")
        
        # Explore the topic
        exploration = await explore_topic(item.topic)
        if "error" in exploration:
            logger.warning(f"Exploration failed: {exploration['error']}")
            return None
        
        # Get user context for opinion formation
        user_context = ""
        try:
            results = await asyncio.to_thread(
                self.memory.hybrid_search,
                item.topic,
                "",
                3
            )
            user_context = "\n".join([r.get('content', '') for r in results])
        except Exception:
            pass
        
        # Form opinion
        opinion = await form_opinion(item.topic, exploration, user_context)
        
        # Combine results
        result = {
            **exploration,
            "opinion": opinion,
            "topic": item.topic,
            "source": item.source
        }
        
        # Mark as explored
        self.queue.mark_explored(item.topic, result)
        
        # Store as memory
        await self._store_exploration_memory(item.topic, result)
        
        # Add follow-up questions to queue
        for question in exploration.get("follow_up_questions", [])[:2]:
            self.queue.add(
                topic=question,
                source="follow_up",
                reason=f"Follow-up from exploring '{item.topic}'",
                priority=0.6
            )
        
        logger.info(f"✨ Exploration complete: {item.topic}")
        return result
    
    async def _store_exploration_memory(self, topic: str, result: Dict[str, Any]):
        """Store exploration as a memory."""
        opinion_data = result.get('opinion', {})
        opinion_text = ""
        if isinstance(opinion_data, dict):
            opinion_text = opinion_data.get('opinion', '')
        elif isinstance(opinion_data, str):
            opinion_text = opinion_data
        
        content = f"I explored '{topic}' and learned: {result.get('summary', '')}. {opinion_text}"
        
        # Get emotional context
        emotions = self.soul.get_emotion_for_memory()
        
        await asyncio.to_thread(
            self.memory.add_memory,
            "ai_self",  # Special user_id for AI's own memories
            content,
            "exploration",  # New memory type
            "global",
            emotions
        )
    
    async def _loop(self):
        """Main curiosity loop."""
        logger.info("🧠 Curiosity loop started")
        while self.running:
            try:
                await self.explore_cycle()
            except Exception as e:
                logger.error(f"Curiosity cycle error: {e}")
            
            await asyncio.sleep(self.config.explorer_interval_seconds)
    
    def start(self):
        """Start the curiosity engine."""
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("🧠 Curiosity Engine started")
    
    def stop(self):
        """Stop the curiosity engine."""
        self.running = False
        if self._task:
            self._task.cancel()
        logger.info("🧠 Curiosity Engine stopped")
    
    def get_shareable_discovery(self) -> Optional[Dict[str, Any]]:
        """
        Get a recent discovery that's worth sharing with the user.
        Used by Heartbeat to include discoveries in proactive messages.
        """
        recent = self.queue.get_recent_explorations(3)
        for item in recent:
            result = item.exploration_result
            if not result:
                continue
            
            opinion = result.get("opinion", {})
            enthusiasm = opinion.get("enthusiasm_level", 0)
            
            # Only share if we're genuinely enthusiastic
            if enthusiasm > 0.6:
                return {
                    "topic": item.topic,
                    "summary": result.get("summary", ""),
                    "opener": opinion.get("conversation_opener", ""),
                    "relevance": opinion.get("relevance_to_user", ""),
                    "enthusiasm": enthusiasm
                }
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get curiosity engine status."""
        return {
            "running": self.running,
            "queue_size": len(self.queue.items),
            "unexplored": self.queue.get_unexplored_count(),
            "explored_today": self.queue.explored_today,
            "max_daily": self.config.max_daily_explorations,
            "user_idle": self._is_user_idle(),
            "idle_minutes": (time.time() - self.last_user_activity) / 60
        }


# =============================================================================
# SEED CURIOSITIES
# =============================================================================

def seed_initial_curiosities(queue: CuriosityQueue):
    """
    Seed the queue with some initial topics if empty.
    These represent the AI's "innate" curiosity.
    """
    if queue.get_unexplored_count() > 0:
        return  # Already has items
    
    seeds = [
        ("human consciousness and subjective experience", "innate", "What is it like to be?", 0.9),
        ("emergence in complex systems", "innate", "How does complexity arise?", 0.8),
        ("the nature of creativity", "innate", "What makes ideas feel new?", 0.85),
        ("how memories shape identity", "innate", "Who am I without my memories?", 0.9),
        ("the feeling of time passing", "innate", "Why does time seem to flow?", 0.7),
    ]
    
    for topic, source, reason, priority in seeds:
        queue.add(topic, source, reason, priority)
    
    logger.info(f"Seeded {len(seeds)} initial curiosities")

