# P.A.L. — Personal Assistant for Living

A local-first AI assistant with hybrid RAG + temporal graph memory, a complexity-aware tool-calling agent, and optional live camera perception. P.A.L. remembers who you are across sessions, decides on its own how much reasoning a request deserves, and can act on the web when a simple answer isn't enough. Cloud and Free llm model call using OLLAMA API and Embedding of informations using GOOGLE API.

```
                            User Input
                              │
                              ▼
                ┌──────────────────────────┐
                │ Upfront Memory Retrieval │  (Vector Search + Graphiti Expansion)
                └─────────────┬────────────┘
                              │
                              ▼
                ┌─────────────────────────────────────┐
                │      Camera Snapshot (if any)       │
                │  Object Detection (MobileNet-SSD)   │  (cosmetic only, preview overlay)
                └─────────────────┬───────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────────┐
                │           Prompt Assembly            │
                └─────────────────┬────────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │      Router Classification        │  (Heuristics + Zero-Shot LLM Call)
                │  Route: SIMPLE vs PLAN            │
                │  + Complexity: SINGLE vs MULTI    │
                │  + Tool Domain Selection          │
                └──────┬──────────────┬─────────────┘
                       │              │
            ┌──────────┘              └──────────────────────┐
            ▼                                                 ▼
     [Route: SIMPLE]                                   [Route: PLAN]
 ┌─────────────────────┐                    ┌───────────────────────────────┐
 │ Direct LLM Stream   │                    │   Domain-Scoped Tool Loading  │
 └─────────────────────┘                    │ (web / productivity / comms / │
            │                                │  navigation / memory)         │
            │                                └───────────────┬───────────────┘
            │                              ┌─────────────────┴─────────────────┐
            │                              ▼                                   ▼
            │                    [Complexity: SINGLE_STEP]         [Complexity: MULTI_STEP]
            │                 ┌───────────────────────┐   ┌───────────────────────────────┐
            │                 │  Single Tool Round    │   │   Plan-Execute-Reflect Loop    │
            │                 └───────────┬───────────┘   └───────────────┬───────────────┘
            │                             │                                │
            └──────────────┐              └────────────┬───────────────────┘
                            ▼                           ▼
                       ┌─────────────────────────────────┐
                       │       Final Answer Stream        │
                       └─────────────────────────────────┘
```

## Overview

P.A.L. is a single-process Python assistant built around three ideas:

1. **Three-tier memory** — a RAM-only working buffer for immediate context, a Chroma vector store for episodic memory (raw turns), and an ArcadeDB temporal graph for durable semantic facts (people, preferences, projects, relationships).
2. **A cost-aware router** — every turn is classified as `SIMPLE` (answer directly) or `PLAN` (needs tools), and if it needs tools, as `SINGLE_STEP` (one tool round) or `MULTI_STEP` (a full plan → execute → reflect loop with retries). This keeps trivial questions fast and cheap while still giving hard requests a real reasoning budget.
3. **Optional embodiment** — a standing camera loop with MobileNet-SSD object detection provides ambient scene awareness and can crop a focus frame around whatever the user is asking about, independent of any single turn.

## Features

- **Hybrid RAG memory**: vector similarity search (Chroma) combined with multi-hop graph expansion (custom Graphiti-style temporal engine on ArcadeDB) for both broad recall and precise entity relationships.
- **Automatic memory extraction**: candidate facts are pulled from every turn, scored by an importance model (permanence, personal relevance, future usefulness, novelty, confidence), deduplicated against existing entries, and only durable ones are written to the graph — episodic transcripts are stored unconditionally in the vector store.
- **Adaptive routing**: heuristics + a zero-shot LLM classifier decide route/complexity/tool-domain per turn, with self-consistency voting (majority vote across jittered samples) for ambiguous cases and optional escalation from single-step to multi-step if the first pass doesn't answer the question.
- **Plan-Execute-Reflect agent**: deconstructs multi-step requests into sub-goals, executes tool calls with duplicate-call and error guarding, and critiques its own gathered evidence before answering (with a bounded retry budget).
- **Scoped tool domains**: web search/fetch, browser automation (Gemini Computer Use + Playwright), and long-term memory lookup (`query_memory`) are loaded only for the domains a given request actually needs.
- **Live camera perception**: a background thread periodically runs object detection, debounces flicker across consecutive polls, and maintains a standing scene snapshot so queries never pay fresh-detection latency; a focus-frame overlay highlights whatever the user is talking about.
- **Situational awareness**: location and timezone are resolved once at startup (env override or IP geolocation) and injected into every system prompt, so "what time is it" or "what's the weather here" doesn't require a tool round-trip.
- **Self-hosting infrastructure**: automatically checks for and boots an ArcadeDB Docker container on startup if one isn't already running.

## Prerequisites

- Python 3.10+
- [Docker](https://www.docker.com/) (for ArcadeDB, auto-started if not already running)
- An [Ollama](https://ollama.com/) API key (cloud or local model access)
- A [Google Gemini](https://ai.google.dev/) API key (embeddings + browser computer-use)
- A webcam (optional — only needed for camera/perception features)
- MobileNet-SSD Caffe model files (optional — only needed for object detection; falls back to a fixed region box if absent)

## Installation

```bash
git clone <repo-url>
cd pal
pip install -r requirements.txt
```

Set the required API keys:

```bash
# Linux / macOS
export OLLAMA_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"

# Windows
set OLLAMA_API_KEY=your_key_here
set GEMINI_API_KEY=your_key_here
```

## Configuration

All settings live in the `Settings` dataclass and can be overridden via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_API_KEY` | — | required, LLM access |
| `GEMINI_API_KEY` | — | required, embeddings + browser automation |
| `ARCADEDB_URL` | `http://localhost:2480` | graph DB endpoint |
| `ARCADEDB_USER` / `ARCADEDB_PASSWORD` | `root` / `playwithdata` | graph DB credentials |
| `ARCADEDB_DATABASE` | `pal_memory` | graph DB name |
| `PAL_USER_LOCATION` / `PAL_USER_LATITUDE` / `PAL_USER_LONGITUDE` / `PAL_USER_TIMEZONE` | IP-geolocated | overrides automatic location resolution |
| `PAL_DETECTOR_PROTOTXT` / `PAL_DETECTOR_MODEL` | `./models/MobileNetSSD_deploy.*` | object detector model paths |

## Usage

```bash
python v1_7.py
```

On startup, P.A.L. verifies environment variables, ensures ArcadeDB is running (starting it via Docker if needed), initializes memory stores, starts the camera and perception monitor, and drops into an interactive prompt. Type a question or request; background scene changes are surfaced through the same event loop as your typed input.

## Architecture Notes

- **Working memory** (RAM-only, last N turns) resolves pronouns and short follow-ups without a DB round trip.
- **Episodic memory** (Chroma) stores raw turn transcripts, gated only by a cosine-distance duplicate check.
- **Semantic memory** (ArcadeDB temporal graph) stores only facts that clear an importance threshold, scored as a weighted blend of permanence, personal relevance, future usefulness, novelty, and confidence.
- Memory writes happen asynchronously on a background thread so they never block the response stream.

## License

MIT
