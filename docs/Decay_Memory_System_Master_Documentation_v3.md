# Decay Memory System

## Human-Like Memory for Large Language Models

### Master Documentation

**Version 10.11**  
**November 2025**

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
   - 1.1 [The Problem: LLM Memory Limitations](#11-the-problem-llm-memory-limitations)
   - 1.2 [The Solution: Decay Memory System](#12-the-solution-decay-memory-system)
   - 1.3 [How It Mimics Human Memory](#13-how-it-mimics-human-memory)
2. [Core Architecture](#2-core-architecture)
   - 2.1 [Technology Stack](#21-technology-stack)
   - 2.2 [Three-Layer Architecture](#22-three-layer-architecture)
   - 2.3 [Memory Types](#23-memory-types)
3. [Feature Details](#3-feature-details)
   - 3.1 [Hybrid Search (Dense + Sparse Vectors)](#31-hybrid-search-dense--sparse-vectors)
   - 3.2 [Memory Decay & Scoring System](#32-memory-decay--scoring-system)
   - 3.3 [Knowledge Graph](#33-knowledge-graph)
   - 3.4 [Project Isolation](#34-project-isolation)
   - 3.5 [Proactive Retrieval](#35-proactive-retrieval)
   - 3.6 [Flight Recorder](#36-flight-recorder)
   - 3.7 [Context Handover](#37-context-handover)
   - 3.8 [Conflict Resolution](#38-conflict-resolution)
   - 3.9 [Memory Feedback System](#39-memory-feedback-system)
   - 3.10 [Fallthrough Search](#310-fallthrough-search)
   - 3.11 [Hot-Reload Configuration System](#311-hot-reload-configuration-system)
   - 3.12 [Heartbeat Service](#312-heartbeat-service)
4. [Version History Highlights](#4-version-history-highlights)
5. [Summary](#5-summary)

---

## 1. Executive Overview

### 1.1 The Problem: LLM Memory Limitations

Large Language Models like Claude face a fundamental limitation: they have no persistent memory between conversations. Each new chat session starts with a blank slate, requiring users to re-explain context, preferences, and project history. This creates significant friction for:

- **Long-term projects** - Progress, decisions, and rationale are lost between sessions
- **Personal interactions** - The AI must re-learn user preferences every conversation
- **Complex workflows** - Multi-session tasks require manual context handoffs

### 1.2 The Solution: Decay Memory System

The Decay Memory System is an MCP (Model Context Protocol) server that provides Claude with human-like memory capabilities. It enables:

- **Persistent storage** of information across chat sessions
- **Intelligent retrieval** using semantic search and keyword matching
- **Natural decay** of memories over time, mimicking human forgetting
- **Project isolation** to keep different work streams separate
- **Seamless continuity** when resuming conversations
- **Proactive agency** through the Heartbeat service (see Section 3.12)

### 1.3 How It Mimics Human Memory

Human memory isn't perfect—we forget things over time, but important or frequently accessed information stays fresh. The Decay Memory System replicates this through:

- **Time-based decay**: Older memories naturally fade, with configurable half-lives per memory type
- **Reinforcement**: Accessed memories get strengthened (+5% per access), just like human recall
- **Recency boost**: Recent memories get temporary priority (1.5x if <24h, 1.2x if <72h)
- **Associative recall**: A knowledge graph connects related concepts for broader retrieval
- **Forgetting**: Deprecated or irrelevant memories fade faster than important ones

---

## 2. Core Architecture

### 2.1 Technology Stack

| Component | Purpose |
|-----------|---------|
| **Python MCP Server** | Implements the Model Context Protocol for Claude integration |
| **Qdrant Vector Database** | Stores memories as high-dimensional vectors for semantic search (runs in Docker) |
| **OpenAI Embeddings** | Converts text to 1536-dimensional vectors (text-embedding-3-small) |
| **Google Gemini** | Powers LLM features like entity extraction, query expansion, and conflict resolution |
| **NetworkX** | Builds and traverses the knowledge graph |

### 2.2 Three-Layer Architecture

The system is built on a robust three-layer architecture:

```
┌─────────────────────────────────────┐
│  Interface Layer: Claude Desktop    │  ← User interface
├─────────────────────────────────────┤
│  Logic Layer: Decay Memory MCP      │  ← Classification, decay, ranking
│  Server (Python)                    │
├─────────────────────────────────────┤
│  Storage Layer: Qdrant Vector DB    │  ← Persistent vector storage
└─────────────────────────────────────┘
```

### 2.3 Memory Types

The system categorizes memories into distinct types, each with its own decay rate and behavior:

| Type | Description | Half-Life | Scope |
|------|-------------|-----------|-------|
| `personal` | Facts about the user | 1 year | Global |
| `preference` | User preferences, style | 6 months | Global |
| `goal` | Ambitions, plans | 3 months | Global |
| `project` | Active work, decisions | 30 days | Project-specific |
| `topic` | Technical knowledge | 1 year | Project-specific |
| `dialog` | Conversation content | 7 days | Project-specific |
| `context_handover` | Session summaries | 30 days | Project-specific |
| `knowledge_graph` | Entity relationships | 6 months | Project-specific |

**Global** memories are available across all projects, while **Project-specific** memories are isolated to their respective projects.

---

## 3. Feature Details

### 3.1 Hybrid Search (Dense + Sparse Vectors)

**Purpose:** Combines semantic understanding with exact keyword matching for optimal retrieval.

**How It Works:**

- **Dense vectors** (1536-dimensional) capture semantic meaning using OpenAI embeddings
- **Sparse vectors** use keyword extraction with TF-IDF-style weighting
- **Stop words filtering** (60+ common words) prevents noise in keyword searches
- **Reciprocal Rank Fusion (RRF)** combines both result sets using Qdrant's native Prefetch+Fusion
- **Safe Mode fallback** - if hybrid search fails, automatically degrades to dense-only search with logging

**Benefit:** Finds memories even when queries use different wording (semantic) while still boosting exact matches (keyword).

### 3.2 Memory Decay & Scoring System

**Purpose:** Simulates natural human forgetting while keeping important memories accessible.

**The Scoring Formula:**

```
Effective Score = Search Score × Decay Factor × Boost Factor × Access Boost
```

**Scoring Components:**

| Component | Formula/Value | Description |
|-----------|---------------|-------------|
| Decay Factor | `0.5^(age / half_life)` | Floor of 0.3 prevents complete disappearance |
| Access Boost | `access_count × 0.05` | +5% per retrieval - frequently used facts become 'stickier' |
| Recency Boost | 1.5x if <24h, 1.2x if <72h | Keeps working memory 'hot' |
| Boost Factor | 1.0x to 5.0x max | Manual boost multiplier for important memories |
| Deprecation Factor | 10x faster decay | For outdated memories |

### 3.3 Knowledge Graph

**Purpose:** Connects related entities and concepts, enabling associative recall.

**How It Works:**

- When memories are stored, Gemini extracts entities (people, projects, concepts) and relationships
- NetworkX builds a directed graph connecting these entities
- Search results can be expanded by following graph connections
- Graph edges also decay over time, weakening old relationships
- Graph cache refreshes every 300 seconds (configurable)

**Available Operations:**

| Operation | Description |
|-----------|-------------|
| `find_connection(a, b)` | Finds the shortest path between two entities |
| `get_related_entities(entity)` | Returns all directly connected entities |
| `get_entity_neighborhood(entity)` | Returns the local subgraph within configurable radius |

### 3.4 Project Isolation

**Purpose:** Keeps different work streams separated while sharing common user information.

**How It Works:**

- Each memory can be tagged with a `project_name`
- `personal`, `preference`, and `goal` types are always **global** (shared across projects)
- `project`, `topic`, and `dialog` types are **project-specific**
- Searches within a project automatically include global memories

**Benefit:** Work on multiple projects without cross-contamination, while maintaining a unified understanding of the user.

### 3.5 Proactive Retrieval

**Purpose:** Automatically surfaces relevant memories without explicit user requests.

**How It Works:**

1. **Trigger Detection** - Analyzes user messages for memory-related signals:
   - Explicit keywords: "remember", "last time", "you mentioned"
   - Entity mentions: Capitalized words suggesting names or projects
   - Project references: "the project", "that issue", "the code"
   - Time-based greetings: "Good morning" after a long gap

2. **Injection Scoring** - Calculates whether retrieved memories should be surfaced based on:
   - Semantic similarity to the current message
   - Decay score (age-adjusted relevance)
   - Memory type weight (project memories prioritized)
   - Configurable injection threshold (default 0.30)

3. **LLM Query Expansion** - Gemini can generate better search queries from ambiguous user messages

4. **Caching** - 60-second cache prevents redundant retrievals

### 3.6 Flight Recorder

**Purpose:** Maintains a detailed log of recent conversation turns for context continuity.

**How It Works:**

- Each conversation turn (user message + AI response) is logged with a timestamp
- Stored in per-project JSONL files for efficient append-only writes
- **Two-phase logging**: `update_session_state()` logs user message, `update_last_turn()` adds AI summary
- Automatic pruning removes entries older than 48 hours or beyond 100 turns
- Asyncio locks prevent race conditions during concurrent access
- Path sanitization prevents path traversal attacks in project names

### 3.7 Context Handover

**Purpose:** Enables seamless continuation between chat sessions.

**How It Works:**

- **At chat start**: `get_context_handover()` retrieves recent flight recorder turns plus the last session summary
- **During chat**: `update_session_state()` and `update_last_turn()` track each conversation turn
- **At chat end**: `finalize_session()` creates a summary, extracts memories, and resets the session

**Format Options:**

| Format | Token Savings | Use Case |
|--------|---------------|----------|
| `condensed` (default) | 60% | Normal operation |
| `full` | 0% | Debugging |
| `minimal` | 80% | Older turns, topics-only |

### 3.8 Conflict Resolution

**Purpose:** Handles contradictory information intelligently, keeping memory consistent.

**How It Works:**

- When adding a memory, the system checks for highly similar existing memories (similarity > 0.85)
- Gemini analyzes the relationship between old and new information
- Resolution types:

| Resolution | Action |
|------------|--------|
| `SUPERSEDE` | New information completely replaces old (old is archived) |
| `UPDATE` | New information modifies/corrects old |
| `COMPLEMENT` | Both are valid; merge into unified memory |
| `UNRELATED` | No conflict; store both |

**Note:** Superseded memories are archived (`type: 'archived_superseded'`) rather than deleted, preserving history.

### 3.9 Memory Feedback System

**Purpose:** Allows explicit feedback to adjust memory importance.

**Available Operations:**

| Operation | Effect |
|-----------|--------|
| `boost_memory(id, reason)` | Increases importance by 1.5x (up to 5x maximum), records history |
| `deprecate_memory(id, reason)` | Multiplies decay rate by 10x, marking as outdated |
| `correct_memory(id, new_content)` | Updates content, regenerates vectors, preserves correction history |
| `delete_memory(id, reason)` | Soft-archives the memory (`type: 'archived_deleted'`), preserving history |

### 3.10 Fallthrough Search

**Purpose:** Expands search when project-specific results are poor.

**How It Works:**

- If the best search result score is below the fallthrough threshold (0.30), the system expands to search all projects
- Results from other projects are marked with `_is_fallthrough` and `_fallthrough_source` metadata
- Uses `expand=False` internally to prevent infinite recursion

**Benefit:** Questions like "What's the gate code?" work transparently regardless of which project the answer is stored in.

### 3.11 Hot-Reload Configuration System

**Purpose:** Allows runtime tuning of all parameters without server restart.

**How It Works:**

- All configurable parameters are stored in `dashboard_config.json`
- Config loader checks file timestamp on each access
- Changes take effect immediately without restart
- Falls back to hardcoded defaults if config file is missing

**Configurable Parameters:**

| Category | Examples |
|----------|----------|
| Decay system | Half-lives per type, decay floor |
| Feedback system | Boost increment, deprecation multiplier, max boost |
| Search config | Fallthrough threshold |
| Knowledge graph | Cache seconds, auto-invalidate |
| Flight recorder | Retention hours, max turns |
| Proactive retrieval | All thresholds and weights |

### 3.12 Heartbeat Service

**Purpose:** Transforms the AI from a reactive assistant into a proactive companion with agency.

**The Problem It Solves:**

Traditional LLM interactions are purely reactive—the AI only responds when invoked. The Heartbeat Service gives the AI companion the ability to initiate contact based on context, creating a more natural, "Her"-style relationship.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    HEARTBEAT ENGINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Context   │  │    Mood     │  │   Memory    │         │
│  │  Gatherers  │  │   Manager   │  │  Retrieval  │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│                ┌─────────────────┐                          │
│                │  LLM Decision   │  ← GPT-4o-mini           │
│                │      Gate       │                          │
│                └────────┬────────┘                          │
│                         │                                   │
│            ┌────────────┼────────────┐                      │
│            ▼            ▼            ▼                      │
│      ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│      │  Toast   │ │   TTS    │ │ Telegram │                │
│      │ Notification│ │ (Future) │ │ (Future) │              │
│      └──────────┘ └──────────┘ └──────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Core Components:**

| Component | Purpose |
|-----------|---------|
| **Context Gatherers** | Collect time, weather, location, and last interaction data |
| **Mood Manager** | Maintains persistent emotional state with circadian rhythm modeling |
| **Memory Retrieval** | Queries decay_memory for contextually relevant memories |
| **LLM Decision Gate** | Cheap model (GPT-4o-mini) decides if outreach is warranted |
| **Delivery Channels** | Windows toast notifications (TTS and Telegram planned) |

**Mood State Model:**

The companion maintains emotional state that evolves over time:

```python
@dataclass
class MoodState:
    current_mood: str       # "neutral", "happy", "concerned"
    energy_level: float     # 0.0 - 1.0, follows circadian rhythm
    intimacy_level: float   # 0.0 - 1.0, grows with positive interactions
    circadian_phase: str    # "morning", "day", "evening", "night"
    consecutive_no_response: int  # Tracks ignored messages
```

**Circadian Energy Model:**

| Time | Phase | Energy Effect |
|------|-------|---------------|
| 6am - 12pm | Morning | +0.1 (rising) |
| 12pm - 6pm | Day | Stable |
| 6pm - 10pm | Evening | -0.05 (winding down) |
| 10pm - 6am | Night | -0.1 (low energy) |

**Constraint System:**

The Heartbeat respects boundaries to avoid being annoying:

| Constraint | Default | Purpose |
|------------|---------|---------|
| Quiet Hours | 10pm - 7am | No messages during sleep |
| Max Daily Messages | 5 | Prevents spam |
| Minimum Gap | 1 hour | Space between messages |
| Decision Threshold | 0.7 | LLM confidence required to speak |

**Security Features (v1.2.1):**

| Feature | Protection |
|---------|------------|
| Atomic File Writes | Prevents state corruption on crash |
| Input Sanitization | Mitigates prompt injection from memories |
| Output Validation | Ensures LLM returns expected schema |
| Strict Timeouts | Prevents zombie processes (15s default) |
| Log Masking | API keys never appear in logs |
| Async Safety | `asyncio.wait_for()` wraps all LLM calls |

**Configuration:**

```python
@dataclass
class HeartbeatConfig:
    interval_seconds: int = 300      # Check every 5 minutes
    min_gap_seconds: int = 3600      # 1 hour between messages
    quiet_hours_start: int = 22      # 10pm
    quiet_hours_end: int = 7         # 7am
    max_daily_messages: int = 5
    decision_threshold: float = 0.7
    llm_provider: str = "openai"     # or "gemini"
    llm_model: str = "gpt-4o-mini"
    companion_name: str = "Her"
```

**Usage:**

```bash
# Run continuously (background service)
python heartbeat.py

# Test single cycle (bypasses constraints)
python heartbeat.py --test
```

**Integration with Decay Memory:**

- Uses `MemorySystem` directly for memory retrieval
- Reads flight recorder to determine last interaction time
- Logs outgoing messages to flight recorder with `[HEARTBEAT]` prefix
- Queries are sanitized before memory search to prevent injection

**Future Roadmap:**

- **TTS Delivery**: Spoken messages via system audio
- **Telegram Integration**: Bidirectional voice messages when away from desktop
- **Calendar Awareness**: "You have a meeting in 30 minutes"
- **Response Detection**: Adjust mood based on whether user responds to heartbeat messages

---

## 4. Version History Highlights

| Version | Changes |
|---------|---------|
| **V10.11** | Heartbeat Service v1.2.1 (proactive agency, mood state, security hardening) |
| **V10.10** | Modular refactoring (13 files, down from single 97KB monolith) |
| **V10.08** | Fixed proactive retrieval thresholds, configurable injection scoring, debug logging |
| **V10.07** | Security hardening (path sanitization), asyncio locks, Python logging, malformed JSON handling |
| **V10.06** | Production hardening, 16-test verification suite passed |
| **V10.04** | Token reduction (60% savings), condensed formats, hot-reload config, Streamlit dashboard |
| **V10.03** | Two-phase flight recorder logging, `update_last_turn` tool |
| **V10.02** | Project isolation leak fix, proactive retrieval score fix |
| **V10.01** | Safe mode fallback for hybrid search failures |
| **V10.00** | Hybrid search (Prefetch+RRF), stop words filtering, graph expansion fix |

---

## 5. Summary

The Decay Memory System transforms Claude from a stateless assistant into a companion that remembers—and now, with the Heartbeat Service, one that can reach out proactively. By combining vector search, knowledge graphs, natural decay, intelligent session management, and autonomous agency, it provides the foundation for truly personalized AI interactions that build on past conversations rather than starting fresh each time.

**Key Differentiators:**

| Feature | Benefit |
|---------|---------|
| **Human-like forgetting** | Important memories persist while trivial ones fade |
| **Reinforcement learning** | Frequently accessed facts become stickier |
| **Seamless continuity** | Resume conversations without re-explaining context |
| **Project organization** | Multiple work streams stay cleanly separated |
| **Proactive awareness** | Relevant memories surface automatically |
| **Autonomous agency** | Heartbeat enables AI-initiated contact |
| **Emotional modeling** | Mood state creates more natural interactions |
| **Token efficiency** | Optimized formats extend conversation length |
| **History preservation** | Soft-delete archiving maintains audit trail |
| **Runtime tuning** | Hot-reload configuration without restarts |
| **Security hardening** | Input sanitization, atomic writes, timeout protection |

---

*Document Version: 3.0*  
*Last Updated: November 2025*  
*Authors: Stephen, Claude & Gemini*
