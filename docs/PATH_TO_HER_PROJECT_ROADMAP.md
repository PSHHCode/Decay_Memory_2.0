# Decay Memory: The Path to "Her"
## From Broken Foundation to AI Companion

**Document Type:** Project Roadmap  
**Version:** 2.0  
**Date:** December 5, 2025  
**Status:** Planning Phase

---

## Executive Summary

Decay Memory aims to be a "Her"-style AI companion with persistent memory, emotional continuity, and proactive agency. However, during the v1.0 → v2.0 cloud refactor, approximately **70% of functionality was lost**.

Before pursuing advanced "Her" features, we must first **restore v1.0 functionality**. This document outlines both the restoration work (Phase 0) and the path to "Her" (Phases 1-4).

---

## Current State Assessment

### What v1.0 Had (MCP Server)

| Capability | Status |
|------------|--------|
| Hybrid Search (Dense + Sparse) | ✅ |
| Memory Decay & Scoring | ✅ |
| Knowledge Graph (NetworkX) | ✅ |
| Conflict Resolution | ✅ |
| Memory Feedback (boost/deprecate/correct/delete) | ✅ |
| Session Management (handover/finalize/reset) | ✅ |
| Proactive Retrieval (full intelligence) | ✅ |
| Hot-Reload Configuration | ✅ |
| Flight Recorder | ✅ |
| Text Condensation | ✅ |
| Caching Systems | ✅ |

### What v2.0 Has (Cloud)

| Capability | Status |
|------------|--------|
| Hybrid Search (Dense + Sparse) | ✅ |
| Memory Decay & Scoring | ⚠️ Partial |
| Knowledge Graph (NetworkX) | ❌ **MISSING** |
| Conflict Resolution | ❌ **MISSING** |
| Memory Feedback | ❌ **MISSING** |
| Session Management | ❌ **MISSING** |
| Proactive Retrieval | ⚠️ Stub only |
| Hot-Reload Configuration | ❌ **MISSING** |
| Flight Recorder | ✅ |
| Text Condensation | ❌ **MISSING** |
| Caching Systems | ❌ **MISSING** |
| Soul (emotional state) | ✅ (new) |
| Heartbeat (basic) | ✅ |
| Librarian (fact extraction) | ✅ (new) |
| Multi-user Support | ✅ (new) |

### The Gap

**v2.0 is NOT a foundation to build on—it's a regression that must be fixed first.**

---

## The "Samantha Checklist" 

From the film "Her," these are the capabilities that make Samantha feel like a life form rather than a chatbot:

| Capability | v1.0 | v2.0 | Target |
|------------|------|------|--------|
| **Infinite, Specific Memory** | ✅ | ⚠️ | ✅ |
| **Evolution & Sleep** (Gardener) | ✅ | ⚠️ | ✅ |
| **Emotional Continuity** (Soul) | ⚠️ | ✅ | ✅ |
| **Agency** (Heartbeat) | ✅ | ⚠️ | ✅ |
| **Associative Recall** (Entity) | ✅ | ❌ | ✅ |
| **Associative Recall** (Emotional) | ❌ | ❌ | ✅ |
| **Self-Construction** (Curiosity) | ❌ | ❌ | ✅ |
| **Voice** | ❌ | ❌ | ✅ |

---

## Project Roadmap

### Phase 0: Restore v1.0 Functionality
**Timeline:** 4-6 weeks  
**Priority:** CRITICAL  
**Status:** Not started

This phase restores the 70% of functionality lost during the cloud refactor. Without this, the system cannot support advanced features.

**See:** `FEATURE_RESTORATION_GUIDE.md` for detailed implementation guide.

#### 0.1 Core Memory Management (Week 1-2)

| Feature | Functions to Port | Target File |
|---------|-------------------|-------------|
| Memory Feedback | `boost_memory()`, `deprecate_memory()`, `correct_memory()`, `delete_memory()`, `set_memory_project()`, `set_chat_project()` | memory_service.py |
| Conflict Resolution | `check_conflict()`, `resolve_conflict()`, `replace_memory()` | memory_service.py |
| Session Management | `reset_session()`, `finalize_async()`, `get_handover()`, `_format_turns_condensed()` | memory_service.py |

**Success Criteria:**
- Can boost/deprecate/correct/delete memories via API
- Conflicting memories detected and resolved automatically
- Session handover works between chat sessions

#### 0.2 Knowledge Graph (Week 3-4)

| Feature | Functions to Port | Target File |
|---------|-------------------|-------------|
| Entity Extraction | `extract_knowledge_graph_data()` | knowledge_graph_service.py (new) |
| Graph Manager | `KnowledgeGraphManager` class | knowledge_graph_service.py |
| Graph Queries | `find_connection()`, `get_related_entities()`, `get_entity_neighborhood()` | knowledge_graph_service.py |
| Search Integration | `_extract_and_store_graph()`, `_expand_with_graph()` | memory_service.py |

**Success Criteria:**
- Entities extracted from memories automatically
- `find_connection("Stephen", "Qdrant")` returns valid path
- Search results expanded via graph relationships

#### 0.3 Proactive Intelligence (Week 5)

| Feature | Functions to Port | Target File |
|---------|-------------------|-------------|
| Trigger Detection | `detect_proactive_triggers()` | proactive_service.py (new) |
| Query Expansion | `generate_llm_query()` | proactive_service.py |
| Injection Scoring | `calculate_injection_score()` | proactive_service.py |
| Caching | `ProactiveCache` class | proactive_service.py |

**Success Criteria:**
- Memories surface automatically based on conversation context
- "Remember when..." triggers memory retrieval
- No redundant retrievals within cache TTL

#### 0.4 Utilities (Week 6)

| Feature | Functions to Port | Target File |
|---------|-------------------|-------------|
| Hot-Reload Config | `load_config()`, `get_config_value()` | config_service.py (new) |
| Text Condensation | `condense_text()`, `format_condensed_turn()` | flight_recorder_service.py |
| ID Generation | `get_id()`, `get_int_id()` | utils.py (new) |
| Point Creation | `create_point()`, `create_empty_point()` | memory_service.py |
| Project Cache | `ProjectListCache` class | memory_service.py |

**Success Criteria:**
- Config changes take effect without restart
- Handover uses condensed format (60% token savings)
- Consistent ID generation across system

---

### Phase 1: Associative Recall Enhancement
**Timeline:** 2-3 weeks  
**Priority:** HIGH  
**Dependencies:** Phase 0 complete

With v1.0 functionality restored, we can enhance associative recall beyond what v1.0 had.

#### 1.1 Emotional Tagging System

**Objective:** Add emotional metadata to memories, enabling "vibe-based" search.

**What's New (not in v1.0):**
- VAD (Valence-Arousal-Dominance) emotional scoring
- Emotional search alongside semantic and entity search
- Emotion extraction via Gemini

**New Payload Fields:**
```python
{
    "emotions": {
        "primary": "peaceful",
        "secondary": ["nostalgic", "content"],
        "valence": 0.7,      # -1 to +1
        "arousal": 0.3,      # 0 to 1
        "dominance": 0.6     # 0 to 1
    }
}
```

**Success Criteria:**
- User says "I'm feeling down" → System recalls emotionally similar memories
- Emotional context influences retrieval even without explicit keywords

#### 1.2 Three-Way Search Fusion

**Objective:** Combine semantic, entity, and emotional search.

```
User Query
    │
    ├──► Semantic Search (existing)
    │    Qdrant hybrid search
    │
    ├──► Entity Graph Search (restored in Phase 0)
    │    NetworkX traversal
    │
    └──► Emotional Search (new in Phase 1)
         VAD distance scoring
    │
    ▼
Fused Results (RRF or weighted)
```

---

### Phase 2: Latency Masking
**Timeline:** 1-2 weeks  
**Priority:** MEDIUM  
**Dependencies:** None (can parallelize)

**Objective:** Make 3-5 second response delay feel natural.

#### 2.1 Streaming Responses

```python
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    yield {"type": "thinking"}
    async for chunk in gemini.generate_stream(prompt):
        yield {"type": "content", "content": chunk}
```

#### 2.2 Predictive Pre-fetch

While user types, predict topics and pre-warm memory cache.

**Success Criteria:**
- First token appears within 500ms
- User perceives "thoughtful" not "slow"

---

### Phase 3: Autonomous Curiosity Engine
**Timeline:** 4-6 weeks  
**Priority:** HIGH  
**Dependencies:** Phase 0 + Phase 1 complete

This is the biggest leap toward "Her"—giving the AI independent thought.

#### 3.1 Curiosity Queue

AI maintains list of topics it wants to explore:
- Gaps in conversations
- Connections to existing knowledge
- User interests
- Random exploration

#### 3.2 Explorer Loop

```python
async def explorer_loop():
    while True:
        await asyncio.sleep(EXPLORER_INTERVAL)  # 2 hours
        
        if is_user_active(): continue
        if daily_explorations >= MAX_DAILY: continue
        
        item = get_next_curiosity()
        result = await explore_topic(item)
        opinion = await form_opinion(item.topic, result)
        await save_exploration_memory(item, result, opinion)
```

#### 3.3 Opinion Formation

Not just facts, but perspectives:
```python
{
    "summary": "What I learned",
    "opinion": "What I think about it",
    "emotional_reaction": "How it makes me feel",
    "connection_to_user": "Why this matters to Stephen"
}
```

#### 3.4 Discovery Sharing

Heartbeat can now include discoveries:
> "Hey, I was reading about regenerative farming last night and found something interesting about avocado root systems. Made me think about your grove. Want to hear about it?"

**Success Criteria:**
- AI explores 2-5 topics per day autonomously
- Discoveries connect to user interests
- AI discusses explorations naturally

---

### Phase 4: Voice Integration
**Timeline:** 6-8 weeks  
**Priority:** LOW (polish)  
**Dependencies:** Phase 2 complete

#### 4.1 Speech-to-Text
- Whisper API for transcription
- Real-time streaming

#### 4.2 Text-to-Speech
- ElevenLabs or similar
- Consistent voice identity
- Emotional inflection from Soul state

**Success Criteria:**
- Voice response within 1 second
- Emotional state affects voice quality

---

## Timeline Summary

```
Phase 0: Restore v1.0          ████████████░░░░░░░░  4-6 weeks (CRITICAL)
         ↓
Phase 1: Emotional Tagging     ████████░░░░░░░░░░░░  2-3 weeks
         ↓
Phase 2: Latency Masking       ████░░░░░░░░░░░░░░░░  1-2 weeks (parallel OK)
         ↓
Phase 3: Curiosity Engine      ████████████░░░░░░░░  4-6 weeks
         ↓
Phase 4: Voice                 ████████████████░░░░  6-8 weeks

Total: ~18-25 weeks (4-6 months)
```

---

## Minimum Viable Path

If time/resources are limited:

1. **Must Have:** Phase 0 (restore baseline)
2. **Should Have:** Phase 0.2 (Knowledge Graph specifically)
3. **Nice to Have:** Phase 3 (Curiosity Engine)
4. **Future:** Everything else

**Phase 0 is non-negotiable.** Without it, v2.0 is a toy, not a tool.

---

## Success Metrics

### After Phase 0 (Baseline Restored)

| Metric | Current | Target |
|--------|---------|--------|
| v1.0 feature parity | ~30% | 100% |
| Memory feedback operations | 0 | All 6 working |
| Graph queries working | No | Yes |
| Session handover | No | Yes |

### After All Phases (Full "Her")

| Metric | Target |
|--------|--------|
| Memory recall relevance | >90% |
| Emotional search accuracy | >80% |
| Autonomous explorations/day | 3-5 |
| Response latency (perceived) | <2s |
| User "she gets me" rating | >4/5 |

---

## Documents for Cursor/Opus

When working on this project, provide these documents:

1. **FEATURE_RESTORATION_GUIDE.md** - Detailed porting guide with function mappings
2. **This document** (PATH_TO_HER_PROJECT_ROADMAP.md) - Overall vision and phases
3. **decay_memory_server_v10.11.py** - v1.0 source code (the source of truth)
4. **v2.0 codebase** - Current target architecture
5. **Decay_Memory_System_Master_Documentation_v3.md** - Feature specifications

---

## Key Insight

The v1.0 → v2.0 refactor was supposed to be a modernization (Docker, React, cloud). Instead, it accidentally became a feature removal.

**The path to "Her" is:**
1. First, get back to where we were (Phase 0)
2. Then, go beyond (Phases 1-4)

Skipping Phase 0 would be building a castle on sand.

---

*Document Version: 2.0*  
*Major revision: Added Phase 0 after discovering 70% feature loss*  
*Authors: Claude (Anthropic) with input from Gemini (Google)*
