

"""
P.A.L. (Personal Assistant for Living) -- Hybrid RAG + Temporal Graph Memory Architecture.


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
                │  tight box on named/located object, │
                │  else region box, else full frame   │
                └─────────────────┬───────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────────┐
                │           Prompt Assembly            │
                │  + Problem-Solving section when the  │
                │    question asks to solve something  │
                │    written/shown in the image        │
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
            │                                                │
            │                              ┌─────────────────┴─────────────────┐
            │                              ▼                                   ▼
            │                    [Complexity: SINGLE_STEP]         [Complexity: MULTI_STEP]
            │                 ┌───────────────────────┐   ┌───────────────────────────────┐
            │                 │  Single Tool Round    │   │   Plan-Execute-Reflect Loop    │
            │                 │  (Max N = 1 Round)    │   │  1. Plan: deconstruct sub-goals│
            │                 └───────────┬───────────┘   │  2. Execute: tool calls, dup-  │
            │                             │                │     call & error guard        │
            │                             │                │  3. Reflect: critique gathered │
            │                             │                │     data, retry if insufficient│
            │                             │                │     (Max N = 3 + 1 retry)     │
            │                             │                └───────────────┬───────────────┘
            │                             │                                │
            │                             │        query_memory tool available mid-loop
            │                             │        for dynamic multi-hop graph lookups
            │                             │                                │
            └──────────────┐              └────────────┬───────────────────┘
                            ▼                           ▼
                       ┌─────────────────────────────────┐
                       │       Final Answer Stream        │
                       └─────────────────────────────────┘

"""

from __future__ import annotations

import asyncio
import enum
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterator, Sequence, TypeVar
from zoneinfo import ZoneInfo

import chromadb
import cv2
import requests
from google import genai
from google.genai import types
from ollama import Client, web_fetch, web_search

# ==============================================================================
# CONFIGURATION & AUTOMATED INITIALIZATION
# ==============================================================================

def ensure_environment_variables() -> None:
    """Verifies that mandatory environment variables exist. Exits early if missing."""
    missing = []
    if not os.environ.get("OLLAMA_API_KEY"):
        missing.append("OLLAMA_API_KEY")
    if not os.environ.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")

    if missing:
        print(f"\n[Initialization Error] Missing required environment variable(s): {', '.join(missing)}", file=sys.stderr)
        print("Please export them before running the script:", file=sys.stderr)
        if os.name == "nt":  # Windows
            for var in missing:
                print(f"  set {var}=your_key_here", file=sys.stderr)
        else:  # Linux / macOS
            for var in missing:
                print(f"  export {var}='your_key_here'", file=sys.stderr)
        sys.exit(1)


def ensure_arcadedb_running(
    arcadedb_url: str = "http://localhost:2480",
    user: str = "root",
    password: str = "playwithdata",
) -> None:
    """Checks if ArcadeDB is running and ready to handle queries.

    If not responsive, attempts to start it via Docker and waits for full engine initialization
    to prevent connection resets/aborts during warm-up.
    """
    print("[Initializer] Checking ArcadeDB availability...")

    def is_engine_ready() -> bool:
        try:
            # Ping the ArcadeDB server root endpoint rather than a non-existent 'system' DB
            res = requests.get(
                f"{arcadedb_url}/api/v1/server",
                auth=(user, password),
                timeout=2,
            )
            return res.status_code in (200, 401, 403)
        except requests.exceptions.RequestException:
            return False

    if is_engine_ready():
        print("[Initializer] ArcadeDB engine is online and ready.")
        return

    print("[Initializer] ArcadeDB service not detected or still booting.")
    print("[Initializer] Attempting to start ArcadeDB via Docker...")

    # 1. Check if docker CLI exists
    try:
        subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[Initialization Failure] Docker CLI is not available in PATH.", file=sys.stderr)
        print("Please start ArcadeDB manually or install Docker.", file=sys.stderr)
        sys.exit(1)

    # 2. Check if container 'arcadedb' exists
    check_cmd = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=arcadedb", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )

    if "arcadedb" in check_cmd.stdout.splitlines():
        print("[Initializer] Starting existing 'arcadedb' container...")
        subprocess.run(["docker", "start", "arcadedb"], check=True)
    else:
        print("[Initializer] Launching new 'arcadedb' container...")
        run_cmd = [
            "docker", "run", "-d",
            "--name", "arcadedb",
            "-p", "2480:2480", 
            "-p", "2424:2424",
            "-e", f"JAVA_OPTS=-Darcadedb.server.rootPassword={password}",
            "arcadedata/arcadedb:latest"
        ]
        subprocess.run(run_cmd, check=True)

    # 3. Poll endpoint until the DB engine responds
    print("[Initializer] Waiting for ArcadeDB engine to finish boot sequence...", end="", flush=True)
    for _ in range(40):
        if is_engine_ready():
            time.sleep(2)  # Buffer for transaction pool readiness
            print("\n[Initializer] ArcadeDB engine fully initialized!")
            return
        print(".", end="", flush=True)
        time.sleep(1)

    print("\n[Initialization Error] ArcadeDB started, but engine timed out during boot.", file=sys.stderr)
    sys.exit(1)


def _require_env(name: str, hint: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"\n[Configuration Error] Missing {name} environment variable.", file=sys.stderr)
        print(hint, file=sys.stderr)
        sys.exit(1)
    return value


@dataclass(frozen=True)
class SamplingOptions:
    temperature: float
    top_p: float
    num_predict: int

    def as_dict(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p, "num_predict": self.num_predict}


@dataclass(frozen=True)
class Settings:
    ollama_api_key: str
    gemini_api_key: str
    model: str = "gemma4:31b-cloud"
    assistant_name: str = "P.A.L."

    # ArcadeDB Configuration
    arcadedb_url: str = os.environ.get("ARCADEDB_URL", "http://localhost:2480")
    arcadedb_user: str = os.environ.get("ARCADEDB_USER", "root")
    arcadedb_password: str = os.environ.get("ARCADEDB_PASSWORD", "playwithdata")
    arcadedb_database: str = os.environ.get("ARCADEDB_DATABASE", "pal_memory")

    # Vector & Deduplication Thresholds
    duplicate_distance_threshold: float = 0.05
    adjudication_distance_threshold: float = 0.35
    max_dynamic_categories: int = 10

    # Sampling Presets
    router_options: SamplingOptions = field(
        default_factory=lambda: SamplingOptions(temperature=0.0, top_p=0.9, num_predict=5)
    )
    # v1.8: the old 5-token router_options budget left zero room for the
    # model to reason before emitting a label — a mid-size model doing a
    # single forward-pass judgment call with no scratch space. This preset
    # is used by the merged route+complexity+domain classifier below, which
    # asks for a short rationale before the label (CoT-before-answer
    # reliably beats zero-shot label-only classification for weaker models).
    router_reasoning_options: SamplingOptions = field(
        default_factory=lambda: SamplingOptions(temperature=0.0, top_p=0.9, num_predict=200)
    )
    # Self-consistency for the genuinely ambiguous middle (the ~20-30% of
    # requests that don't hit a fast-path keyword/pattern): fire the merged
    # classifier this many times, at slightly jittered temperature, and take
    # a majority vote instead of trusting a single determinstic sample.
    router_self_consistency_samples: int = 3
    router_self_consistency_temp_jitter: float = 0.15
    # If a SINGLE_STEP tool round turns out not to have actually answered
    # the request, escalate to the full plan->execute->reflect loop instead
    # of returning an incomplete answer with no recovery path.
    single_step_escalate_to_multi_step: bool = True
    # Optional append-only log of fast-path hits and self-consistency vote
    # splits, so keyword heuristic tuning can be evidence-based instead of
    # guesswork. None disables file logging (console logging still happens).
    routing_log_path: str | None = None
    tool_orchestration_options: SamplingOptions = field(
        default_factory=lambda: SamplingOptions(temperature=0.2, top_p=0.9, num_predict=2048)
    )
    final_answer_options: SamplingOptions = field(
        default_factory=lambda: SamplingOptions(temperature=0.6, top_p=0.95, num_predict=1024)
    )
    # Dedicated preset for lightweight NER on the raw user query at read-time.
    # Needs more headroom than router_options (num_predict=5) but far less
    # than tool_orchestration_options; a bare list of names is short JSON.
    entity_extraction_options: SamplingOptions = field(
        default_factory=lambda: SamplingOptions(temperature=0.0, top_p=0.9, num_predict=256)
    )
    # Used for both the plan-deconstruction step and the post-loop
    # quality critique. Needs more headroom than the router's 5-token
    # budget (a sub-goal list or a critique verdict is a few sentences)
    # but stays well short of a full tool-orchestration turn.
    planning_options: SamplingOptions = field(
        default_factory=lambda: SamplingOptions(temperature=0.1, top_p=0.9, num_predict=512)
    )

    # Round budgets are now tiered by RouteComplexity instead of one flat
    # constant applied to every PLAN turn regardless of how complex the
    # request actually is.
    single_step_max_rounds: int = 1
    max_agent_tool_rounds: int = 3
    max_reflection_retries: int = 1
    curiosity_threshold: float = 0.85
    curiosity_question_cooldown_turns: int = 2

    chroma_path: str = "./my_chroma_db"
    category_registry_name: str = "category_registry"

    # Focus-frame object detector (MobileNet-SSD, Caffe). Optional: if the
    # files aren't present at these paths, detection is skipped and the
    # focus-frame falls back to a fixed region box — see ObjectDetector.
    object_detector_prototxt_path: str = os.environ.get(
        "PAL_DETECTOR_PROTOTXT", "./models/MobileNetSSD_deploy.prototxt"
    )
    object_detector_model_path: str = os.environ.get(
        "PAL_DETECTOR_MODEL", "./models/MobileNetSSD_deploy.caffemodel"
    )
    object_detector_min_confidence: float = 0.45

    # --------------------------------------------------------------------
    # Web Interaction (Gemini Computer Use + Playwright)
    # --------------------------------------------------------------------
    # Uses Google Gemini's Computer Use model with Playwright to interact
    # with websites (click, type, scroll, navigate). The model sees
    # screenshots and returns actions; Playwright executes them client-side.
    browser_interaction_model: str = "gemini-3.5-flash"
    browser_viewport_width: int = 1440
    browser_viewport_height: int = 900
    browser_max_steps: int = 25
    browser_headless: bool = False
    browser_safety_policy: str = "DEFAULT"
    # Initial URL to navigate to when none is specified by the task.
    browser_default_url: str = "https://www.google.com"

    # --------------------------------------------------------------------
    # v1.9: Continuous perception (standing scene monitor)
    # --------------------------------------------------------------------
    # Previously, object detection only ever ran inside a user turn
    # (set_focus_target, triggered from handle_turn). The camera thread
    # itself always ran continuously (raw frame grab), but *understanding*
    # of the scene was purely reactive — nothing was watching between
    # turns. This block enables a second background loop that periodically
    # runs detection independent of any query and maintains a standing
    # PerceptionSnapshot, so (a) a query never pays fresh detection latency
    # and (b) a sustained scene change can trigger a proactive turn.
    enable_continuous_perception: bool = True
    perception_poll_interval_seconds: float = 2.0
    # A detected change must persist across this many consecutive polls
    # before it's treated as real (debounces flicker/misdetections from a
    # single noisy frame) — see CameraManager._perception_loop.
    perception_stability_polls: int = 2
    # Floor between two proactive, unprompted scene comments, regardless of
    # how many stable changes occur in between — an ambient assistant that
    # narrates every object it sees is worse than a silent one.
    proactive_scene_cooldown_seconds: float = 120.0

    # Shared taxonomy: identical labels are used as Chroma collection names
    # AND as the `category` property on graph vertices/edges. This is what
    # lets a vector hit's metadata point directly at a graph scope instead
    # of re-deriving it heuristically at query time.
    memory_categories: tuple[str, ...] = ("profile", "conversations", "visual_memories", "projects")
    entity_types: tuple[str, ...] = (
        "PERSON", "OBJECT", "LOCATION", "PREFERENCE", "ORGANIZATION", "EVENT", "TOPIC", "OTHER"
    )

    # --------------------------------------------------------------------
    # Semantic Memory Pipeline (graph write-path only)
    # --------------------------------------------------------------------
    # The Graph DB is long-term SEMANTIC memory: durable facts about the
    # world/user only. Episodic content (raw turns, transcripts) stays in
    # the Vector DB, written unconditionally in extract_and_store; it never
    # reaches this gate. Only candidates classified into one of the types
    # below, scoring above the threshold, survive to become graph edges.
    semantic_fact_types: tuple[str, ...] = (
        "Person", "Organization", "Place", "Project", "Skill", "Profession",
        "Preference", "Habit", "Goal", "Relationship", "Device", "Language", "Interest",
    )
    # 0.30*Permanence + 0.25*PersonalRelevance + 0.20*FutureUsefulness
    # + 0.15*Novelty + 0.10*Confidence
    semantic_importance_weights: dict[str, float] = field(
        default_factory=lambda: {
            "permanence": 0.30,
            "personal_relevance": 0.25,
            "future_usefulness": 0.20,
            "novelty": 0.15,
            "confidence": 0.10,
        }
    )
    semantic_importance_threshold: float = 0.55
    # "Will this fact still be worth recalling roughly a year from now?"
    # gate, separate from the generic 0-1 `future_usefulness` score used
    # in the importance-weighting formula above. `future_usefulness`
    # answers "would recalling this help a later turn" in the abstract;
    # `semantic_long_term_horizon_days` gives that judgment a concrete,
    # checkable time horizon so a fact can be scored as durable enough to
    # matter near-term (e.g. "user is tired today") without also being
    # treated as something worth carrying for a year (e.g. "user's
    # profession").
    semantic_long_term_horizon_days: int = 365
    semantic_long_term_min_permanence: float = 0.6

    # --------------------------------------------------------------------
    # Episodic Memory (Chroma vector-store write-path)
    # --------------------------------------------------------------------
    # Below this distance, a new turn is treated as a near-duplicate of
    # an already-stored one and is skipped (logged as
    # "[VectorDB unchanged]") rather than written again.
    #
    # BUGFIX (was 0.02, calibrated for squared-L2): collections are now
    # explicitly created with hnsw:space="cosine" (see CategoryRegistry),
    # so this threshold must be read as a COSINE distance: 0 = identical
    # direction, 1 = orthogonal, 2 = opposite. The old value silently
    # assumed embeddings were unit-normalized L2 vectors, which
    # gemini-embedding-001 does not guarantee -- so near-identical
    # sentences ("user is drinking coffee" stated twice) could sit well
    # above 0.02 in raw L2 space and never trigger the duplicate gate at
    # all. 0.05 in cosine-distance space reliably catches paraphrases /
    # repeats of the same statement while still treating genuinely
    # different content (even on a related topic) as new.
    episodic_duplicate_distance_threshold: float = 0.05

    # --------------------------------------------------------------------
    # Working Memory (Layer 1 of the three-tier architecture)
    # --------------------------------------------------------------------
    # Verbatim recent-turn buffer, kept in RAM only -- never written to
    # Chroma or ArcadeDB. This is what lets the model resolve "it"/"that"
    # and short follow-ups without a DB round trip; Episodic (Chroma) and
    # Semantic (ArcadeDB) memory remain the long-term stores below it.
    working_memory_max_turns: int = 5

    # Situational awareness: resolved once at startup (env override, else
    # IP-based geolocation, else "Unknown") and injected into every system
    # prompt so direct questions like "what time is it" or "what's the
    # weather here" don't need a tool round-trip or guesswork.
    user_location_name: str = "Unknown"
    user_latitude: float | None = None
    user_longitude: float | None = None
    user_timezone: str | None = None

    @staticmethod
    def _resolve_location() -> tuple[str, float | None, float | None, str | None]:
        env_name = os.environ.get("PAL_USER_LOCATION")
        if env_name:
            lat = os.environ.get("PAL_USER_LATITUDE")
            lon = os.environ.get("PAL_USER_LONGITUDE")
            tz = os.environ.get("PAL_USER_TIMEZONE")
            return env_name, (float(lat) if lat else None), (float(lon) if lon else None), tz

        # ip-api.com free tier: HTTP only (no SSL on the free endpoint),
        # no API key, 45 requests/minute cap — fine for a single call at
        # startup. `fields=` trims the response to what we actually use.
        try:
            res = requests.get(
                "http://ip-api.com/json/?fields=status,message,city,regionName,country,lat,lon,timezone",
                timeout=3,
            )
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "success":
                    name = ", ".join(p for p in (data.get("city"), data.get("country")) if p) or "Unknown"
                    return name, data.get("lat"), data.get("lon"), data.get("timezone")
                print(f"[Initializer] ip-api.com lookup returned failure: {data.get('message')}")
        except Exception as exc:
            print(f"[Initializer] IP-based location lookup failed, defaulting to 'Unknown': {exc}")
        return "Unknown", None, None, None

    @classmethod
    def from_env(cls) -> "Settings":
        ollama_key = _require_env("OLLAMA_API_KEY", "set OLLAMA_API_KEY=your_key")
        gemini_key = _require_env("GEMINI_API_KEY", "set GEMINI_API_KEY=your_key")
        location_name, lat, lon, tz = cls._resolve_location()
        return cls(
            ollama_api_key=ollama_key,
            gemini_api_key=gemini_key,
            user_location_name=location_name,
            user_latitude=lat,
            user_longitude=lon,
            user_timezone=tz,
        )


# ==============================================================================
# TEXT CLEANING HELPERS
# ==============================================================================

LEAKED_REASONING_PATTERN = re.compile(r"<\|?/?(?:tool_call|channel|think(?:ing)?)[^>]*\|?>")
_FIRST_WORD_REPEAT_PATTERN = re.compile(r"^(\S+)\s+\1\b")


def strip_leaked_reasoning(text: str) -> str:
    return LEAKED_REASONING_PATTERN.sub("", text).strip()


def dedupe_leading_repeated_word(text: str) -> str:
    match = _FIRST_WORD_REPEAT_PATTERN.match(text)
    if not match:
        return text
    return text[len(match.group(1)):].lstrip()


def clean_model_output(raw_text: str) -> str:
    return dedupe_leading_repeated_word(strip_leaked_reasoning(raw_text))


# ==============================================================================
# LLM & EMBEDDING CLIENTS
# ==============================================================================

@dataclass
class ChatResult:
    raw_message: Any
    cleaned_content: str


class LLMClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = Client(
            host="https://ollama.com",
            headers={
                "Authorization": "Bearer " + settings.ollama_api_key,
                "Content-Type": "application/json",
            },
        )

    @property
    def raw_client(self) -> Client:
        return self._client

    def chat_once(
        self,
        messages: Sequence[dict],
        options: SamplingOptions,
        tools: Sequence | None = None,
        think: bool = False,
    ) -> ChatResult:
        response = self._client.chat(
            model=self._settings.model,
            messages=messages,
            tools=tools,
            stream=False,
            think=think,
            options=options.as_dict(),
        )
        raw_content = response["message"]["content"] if response["message"].get("content") else ""
        return ChatResult(raw_message=response.message, cleaned_content=clean_model_output(raw_content))

    def chat_stream(self, messages: Sequence[dict], options: SamplingOptions) -> Iterator[dict]:
        return self._client.chat(
            model=self._settings.model,
            messages=messages,
            stream=True,
            think=False,
            options=options.as_dict(),
        )


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float] | None:
        try:
            response = self._client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return response.embeddings[0].values
        except Exception as exc:
            print(f"[Embedding Error] : {exc}")
            return None


# ==============================================================================
# ARCADEDB GRAPH CLIENT & TEMPORAL GRAPHITI INTEGRATION
# ==============================================================================

class ArcadeDBClient:
    """REST Client for ArcadeDB Graph Engine storing spatial & temporal facts."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._base_url = settings.arcadedb_url
        self._auth = (settings.arcadedb_user, settings.arcadedb_password)
        self._db = settings.arcadedb_database
        self._init_database()

    def _init_database(self) -> None:
        """Ensure ArcadeDB database and schema classes (Vertex/Edge) exist."""
        try:
            requests.post(
                f"{self._base_url}/api/v1/server",
                auth=self._auth,
                json={"command": f"create database {self._db}"},
                timeout=5,
            )
        except Exception:
            pass  # DB might already exist

        # Create Graph Types: Entity (Vertex), RELATES_TO (Edge)
        # Schema v2: entities are typed (PERSON/OBJECT/LOCATION/...) and
        # tagged with the same category taxonomy used by the Chroma
        # collections, so a vector hit's metadata scopes the graph
        # traversal directly instead of guessing via keyword overlap.
        self.query_cypher("CREATE VERTEX TYPE Entity IF NOT EXISTS")
        self.query_cypher("CREATE PROPERTY Entity.name IF NOT EXISTS STRING")          # canonical, lowercased key
        self.query_cypher("CREATE PROPERTY Entity.display_name IF NOT EXISTS STRING")  # original casing for output
        self.query_cypher("CREATE PROPERTY Entity.entity_type IF NOT EXISTS STRING")   # PERSON/OBJECT/LOCATION/...
        self.query_cypher("CREATE PROPERTY Entity.category IF NOT EXISTS STRING")      # profile/conversations/...
        self.query_cypher("CREATE INDEX ON Entity (name) UNIQUE")

        self.query_cypher("CREATE EDGE TYPE RELATES_TO IF NOT EXISTS")
        self.query_cypher("CREATE PROPERTY RELATES_TO.relation IF NOT EXISTS STRING")
        self.query_cypher("CREATE PROPERTY RELATES_TO.valid_from IF NOT EXISTS STRING")
        self.query_cypher("CREATE PROPERTY RELATES_TO.valid_to IF NOT EXISTS STRING")
        self.query_cypher("CREATE PROPERTY RELATES_TO.source_type IF NOT EXISTS STRING")  # text or image
        self.query_cypher("CREATE PROPERTY RELATES_TO.category IF NOT EXISTS STRING")      # inherited from the memory entry
        self.query_cypher("CREATE PROPERTY RELATES_TO.confidence IF NOT EXISTS FLOAT")     # extraction confidence, 0-1
        self.query_cypher("CREATE PROPERTY RELATES_TO.semantic_type IF NOT EXISTS STRING")     # Person/Preference/Skill/...
        self.query_cypher("CREATE PROPERTY RELATES_TO.importance_score IF NOT EXISTS FLOAT")   # weighted score, 0-1

    def query_cypher(self, command: str, params: dict | None = None) -> list[dict]:
        url = f"{self._base_url}/api/v1/command/{self._db}"
        payload = {"language": "cypher", "command": command}
        if params:
            payload["params"] = params
        try:
            res = requests.post(url, auth=self._auth, json=payload, timeout=8)
            if res.status_code == 200:
                return res.json().get("result", [])
        except Exception as exc:
            print(f"[ArcadeDB Query Error] : {exc}")
        return []


_STOPWORDS = frozenset(
    "the a an and or but if then else when where what who how why is are was were "
    "be been being have has had do does did will would could should may might must "
    "can this that these those there here about into over under again further "
    "you your yours i me my mine we us our they them their he she him her its it "
    "for with from not too very just also".split()
)


def _parse_json_array(raw_text: str) -> list | None:
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else None
    except Exception as exc:
        print(f"[Graph Extraction Error] Parse failed: {exc} | raw={raw!r}")
        return None


def _parse_json_object(raw_text: str) -> dict | None:
    """Same fence-stripping/parse robustness as _parse_json_array, but
    for single-object responses (e.g. classifier verdicts like
    {"worth_storing": true}) rather than candidate lists."""
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        print(f"[Episodic Worthiness Error] Parse failed: {exc} | raw={raw!r}")
        return None


def _coerce_enum(value: Any, allowed: Sequence[str], default: str) -> str:
    candidate = str(value).strip().lower() if value else ""
    for option in allowed:
        if option.lower() == candidate:
            return option
    return default


def _coerce_enum_optional(value: Any, allowed: Sequence[str]) -> str | None:
    """Like _coerce_enum but returns None instead of a default when there's
    no match -- used where "doesn't fit any allowed type" must mean
    "discard the candidate", not "silently file it under some fallback"."""
    candidate = str(value).strip().lower() if value else ""
    for option in allowed:
        if option.lower() == candidate:
            return option
    return None


def _clamp01(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class MemoryCandidate:
    """A single candidate fact after extraction + semantic classification,
    prior to the importance-scoring and duplicate-detection gates."""
    subject: str
    subject_display: str
    subject_type: str
    relation: str
    object: str
    object_display: str
    object_type: str
    semantic_type: str
    category: str
    permanence: float
    personal_relevance: float
    future_usefulness: float
    novelty: float
    confidence: float
    importance_score: float = 0.0
    # Explicit 1-year usefulness verdict, filled in by ImportanceScorer:
    # "durable enough that it should still be worth recalling roughly
    # `settings.semantic_long_term_horizon_days` from now?" Distinct from
    # the raw `future_usefulness` 0-1 score above, which is an input to
    # the weighted importance formula, not a durability verdict itself.
    is_long_term_useful: bool = False


class MemoryCandidateExtractor:
    """Pipeline stage 1+2: Memory Candidate Extraction + Semantic Memory
    Classifier.

    Extraction is treated as a reasoning task, not a parsing task: the
    prompt asks the model to judge, per candidate, whether the fact will
    still improve responses months from now, and to only keep it if it
    fits one of the allowed durable semantic types (Person, Preference,
    Skill, Goal, ...). Conversation turns, greetings, questions, and
    one-time/camera-frame observations have no matching semantic type and
    are expected to be omitted by the model rather than filtered
    afterwards by keyword -- a regex cannot tell "I started learning
    Graphiti yesterday" (-> USER LEARNS Graphiti) apart from "what is
    Docker?" (discard), but the reasoning step can.
    """

    def __init__(self, llm_client: LLMClient, settings: Settings):
        self._llm = llm_client
        self._settings = settings

    def extract(self, text_content: str, default_category: str) -> list[MemoryCandidate]:
        semantic_types = ", ".join(self._settings.semantic_fact_types)
        entity_types = ", ".join(self._settings.entity_types)
        categories = ", ".join(self._settings.memory_categories)
        prompt = (
            "You extract durable, long-term SEMANTIC knowledge for a knowledge graph -- "
            "not a transcript of the conversation. For every candidate fact, ask: "
            "'Will this still improve responses months from now?' Keep it only if yes, "
            "and only if it fits one of the allowed semantic types below. Infer durable "
            "facts instead of storing literal dialogue, e.g. 'I started learning Graphiti "
            "yesterday' becomes (user, LEARNS, graphiti), not (user, STARTED_LEARNING, yesterday).\n\n"
            f"Observation: '{text_content}'\n\n"
            "Do NOT emit facts for: greetings, questions, temporary observations, camera "
            "frames, conversation metadata, assistant responses, or one-time statements.\n\n"
            "Return ONLY a JSON array, no prose, no markdown fences. Each element:\n"
            '{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", '
            '"object_type": "...", "semantic_type": "...", "category": "...", '
            '"permanence": 0-1, "personal_relevance": 0-1, "future_usefulness": 0-1, '
            '"novelty": 0-1, "confidence": 0-1}\n'
            f"subject_type/object_type must be one of: {entity_types}.\n"
            f"semantic_type must be one of: {semantic_types}. If nothing fits, omit the "
            "fact entirely rather than guessing.\n"
            f"category must be one of: {categories} (default '{default_category}' if unsure).\n"
            "relation must be a short UPPER_SNAKE_CASE verb phrase.\n"
            "Score each field 0.0-1.0: permanence (how long-lived the fact is), "
            "personal_relevance (how specific to this user), future_usefulness (would "
            "recalling it help a later turn), novelty (not already obvious or already "
            "restated), confidence (how sure you are it's factually correct).\n"
            'Example: [{"subject": "user", "subject_type": "PERSON", "relation": "LEARNS", '
            '"object": "graphiti", "object_type": "TOPIC", "semantic_type": "Skill", '
            '"category": "projects", "permanence": 0.8, "personal_relevance": 0.9, '
            '"future_usefulness": 0.85, "novelty": 0.7, "confidence": 0.9}]\n'
            "If no durable fact is present, return []."
        )
        res = self._llm.chat_once(
            messages=[{"role": "user", "content": prompt}],
            options=self._settings.tool_orchestration_options,
        )

        raw = _parse_json_array(res.cleaned_content)
        if not raw:
            return []

        candidates: list[MemoryCandidate] = []
        for item in raw:
            sub_raw = str(item.get("subject", "")).strip()
            obj_raw = str(item.get("object", "")).strip()
            sub = sub_raw.lower()
            obj = obj_raw.lower()
            rel = str(item.get("relation", "")).upper().strip().replace(" ", "_")
            semantic_type = _coerce_enum_optional(item.get("semantic_type"), self._settings.semantic_fact_types)
            if not (sub and rel and obj and semantic_type):
                # Missing pieces, or the classifier judged this isn't durable
                # semantic knowledge -- discard rather than store as-is.
                continue

            candidates.append(
                MemoryCandidate(
                    subject=sub,
                    subject_display=sub_raw,
                    subject_type=_coerce_enum(item.get("subject_type"), self._settings.entity_types, "OTHER"),
                    relation=rel,
                    object=obj,
                    object_display=obj_raw,
                    object_type=_coerce_enum(item.get("object_type"), self._settings.entity_types, "OTHER"),
                    semantic_type=semantic_type,
                    category=_coerce_enum(item.get("category"), self._settings.memory_categories, default_category),
                    permanence=_clamp01(item.get("permanence")),
                    personal_relevance=_clamp01(item.get("personal_relevance")),
                    future_usefulness=_clamp01(item.get("future_usefulness")),
                    novelty=_clamp01(item.get("novelty")),
                    confidence=_clamp01(item.get("confidence")),
                )
            )
        return candidates


class ImportanceScorer:
    """Pipeline stage 3: Importance Scoring. Weighted gate; stores are
    only permanent-memory-worthy above `settings.semantic_importance_threshold`.
    Weights and threshold are configurable via Settings, not hardcoded here.

    This stage also renders the explicit "will this still be useful in
    ~a year?" verdict (`is_long_term_useful`) that the weighted score
    alone doesn't give you: `importance_score` blends five 0-1 inputs
    into one number, so a fact can clear the importance threshold on the
    strength of e.g. high novelty + confidence while actually being
    short-lived (low permanence). The long-term gate looks at permanence
    specifically, against a concrete time horizon, rather than folding
    that judgment back into the same blended score.
    """

    def __init__(self, settings: Settings):
        self._settings = settings

    def filter(self, candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
        weights = self._settings.semantic_importance_weights
        threshold = self._settings.semantic_importance_threshold
        min_permanence = self._settings.semantic_long_term_min_permanence
        kept: list[MemoryCandidate] = []
        for c in candidates:
            c.importance_score = (
                weights.get("permanence", 0.0) * c.permanence
                + weights.get("personal_relevance", 0.0) * c.personal_relevance
                + weights.get("future_usefulness", 0.0) * c.future_usefulness
                + weights.get("novelty", 0.0) * c.novelty
                + weights.get("confidence", 0.0) * c.confidence
            )
            # Explicit ~1-year usefulness evaluation: permanence is the
            # field that actually speaks to durability over a long
            # horizon, so gate on it directly rather than inferring it
            # from the blended importance_score.
            c.is_long_term_useful = c.permanence >= min_permanence
            if c.importance_score >= threshold:
                kept.append(c)
            elif c.is_long_term_useful and c.importance_score >= threshold - 0.05:
                # Near-miss on the blended score, but independently
                # judged durable for the long term (~1 year out) -- let
                # the explicit durability signal rescue it rather than
                # silently dropping a fact that will likely still
                # matter, purely because one of the other four inputs
                # (novelty/confidence/personal_relevance/future_usefulness)
                # happened to score low this one time.
                kept.append(c)
        return kept


class DuplicateDetector:
    """Pipeline stage 4: Duplicate Detection, run just before the GraphDB
    write. An exact (subject, relation, object) match is a duplicate --
    refreshed in place instead of re-inserted. A (subject, relation) match
    against a different object is a state change -- the old edge is
    temporally invalidated (Graphiti-style), not left standing alongside
    the new one as if both were simultaneously true."""

    def __init__(self, arcadedb: ArcadeDBClient):
        self._db = arcadedb

    _SCORE_EPSILON = 1e-6

    def resolve(self, candidate: MemoryCandidate, timestamp_str: str) -> str:
        """Returns one of:
          - 'unchanged': exact (subject, relation, object) match already
            in the graph, AND confidence/importance are (near-)identical
            -- genuinely nothing to write, no update query issued.
          - 'duplicate': exact (subject, relation, object) match, but
            confidence/importance moved -- edge refreshed in place.
          - 'new': no exact match (either a brand-new fact, or a state
            change under the same (subject, relation) against a
            different object) -- caller should create a new edge; any
            prior edge for the same (subject, relation) is temporally
            invalidated first.

        Distinguishing 'unchanged' from 'duplicate' is what lets the
        caller print an honest "[GraphDB unchanged]" instead of claiming
        "[GraphDB updated]" on a turn where re-stating an already-known
        fact triggered zero actual writes.
        """
        exact = self._db.query_cypher(
            "MATCH (s:Entity {name: $sub})-[r:RELATES_TO {relation: $rel}]->(o:Entity {name: $obj}) "
            "WHERE r.valid_to IS NULL RETURN r LIMIT 1",
            {"sub": candidate.subject, "rel": candidate.relation, "obj": candidate.object},
        )
        if exact:
            existing = exact[0].get("r", {}) if isinstance(exact[0], dict) else {}
            existing_confidence = existing.get("confidence")
            existing_score = existing.get("importance_score")
            unchanged = (
                existing_confidence is not None
                and existing_score is not None
                and abs(float(existing_confidence) - candidate.confidence) < self._SCORE_EPSILON
                and abs(float(existing_score) - candidate.importance_score) < self._SCORE_EPSILON
            )
            if unchanged:
                return "unchanged"

            self._db.query_cypher(
                "MATCH (s:Entity {name: $sub})-[r:RELATES_TO {relation: $rel}]->(o:Entity {name: $obj}) "
                "WHERE r.valid_to IS NULL "
                "SET r.confidence = $confidence, r.importance_score = $score",
                {
                    "sub": candidate.subject, "rel": candidate.relation, "obj": candidate.object,
                    "confidence": candidate.confidence, "score": candidate.importance_score,
                },
            )
            return "duplicate"

        # Different object under the same (subject, relation): supersede,
        # don't duplicate -- this is Graphiti's temporal-invalidation move.
        self._db.query_cypher(
            "MATCH (s:Entity {name: $sub})-[r:RELATES_TO {relation: $rel}]->(o:Entity) "
            "WHERE r.valid_to IS NULL "
            "SET r.valid_to = $timestamp",
            {"sub": candidate.subject, "rel": candidate.relation, "timestamp": timestamp_str},
        )
        return "new"


class GraphitiTemporalEngine:
    """Uses Graphiti semantics to convert observations (text/image) into
    temporal graph nodes/edges with validity periods (valid_from/valid_to).

    Schema v2 storage model
    ------------------------
    Every Entity vertex carries `entity_type` (PERSON/OBJECT/LOCATION/
    PREFERENCE/ORGANIZATION/EVENT/TOPIC/OTHER) and `category`, the latter
    drawn from the exact same taxonomy as the Chroma collections
    (`settings.memory_categories`). Every RELATES_TO edge inherits that
    same `category`. This is what makes Filter 1 -> Filter 2 a structural
    handoff instead of a keyword guess: a vector hit's stored metadata
    names its entities directly, and those names carry a category that
    scopes the graph traversal to the same domain the hit came from.
    """

    def __init__(self, arcadedb: ArcadeDBClient, llm_client: LLMClient, settings: Settings):
        self._db = arcadedb
        self._llm = llm_client
        self._settings = settings
        # Semantic memory write-path pipeline (Required Pipeline from the
        # redesign spec): Candidate Extraction -> Semantic Classifier ->
        # Importance Scoring -> Duplicate Detection -> GraphDB. Each stage
        # is its own class so thresholds/weights/extraction prompts can be
        # tuned or swapped independently.
        self._extractor = MemoryCandidateExtractor(llm_client, settings)
        self._scorer = ImportanceScorer(settings)
        self._dedup = DuplicateDetector(arcadedb)

    # ------------------------------------------------------------------
    # WRITE PATH
    # ------------------------------------------------------------------
    def extract_and_upsert_facts(
        self, text_content: str, timestamp_str: str, default_category: str, source_type: str = "text"
    ) -> list[dict]:
        """Runs the semantic memory pipeline and upserts only what survives
        every gate, with Graphiti-style temporal invalidation.

        Conversation logging is NOT this method's job: episodic content is
        written unconditionally to the Vector DB by the caller regardless
        of what (if anything) makes it into the graph here. This is what
        keeps the Graph DB a semantic long-term memory instead of a
        transcript -- most turns ("what is Docker?", "hello") will
        legitimately produce an empty result below.

        Returns the canonicalized fact list (subject/relation/object/
        category) for facts that now exist as edges -- new or refreshed --
        so the caller can tag the Chroma document with the same entity
        names, closing the loop between vector store and graph.
        """
        # Stage 1+2: extraction + semantic classification
        candidates = self._extractor.extract(text_content, default_category)
        # Stage 3: importance scoring gate
        scored = self._scorer.filter(candidates)

        canonical_facts: list[dict] = []
        for c in scored:
            # Stage 4: duplicate detection (also handles temporal
            # invalidation of a superseded state for the same subject+relation)
            outcome = self._dedup.resolve(c, timestamp_str)

            # Stage 5: GraphDB write -- only create a new edge if this
            # wasn't resolved as an unchanged or refreshed duplicate above.
            if outcome == "new":
                self._db.query_cypher(
                    "MERGE (s:Entity {name: $sub}) "
                    "SET s.display_name = $sub_display, s.entity_type = $sub_type, s.category = $category "
                    "MERGE (o:Entity {name: $obj}) "
                    "SET o.display_name = $obj_display, o.entity_type = $obj_type, o.category = $category "
                    "CREATE (s)-[r:RELATES_TO {"
                    "  relation: $rel, valid_from: $timestamp, valid_to: null, "
                    "  source_type: $source_type, category: $category, semantic_type: $semantic_type, "
                    "  confidence: $confidence, importance_score: $score, "
                    "  long_term_useful: $long_term_useful"
                    "}]->(o)",
                    {
                        "sub": c.subject, "sub_display": c.subject_display, "sub_type": c.subject_type,
                        "obj": c.object, "obj_display": c.object_display, "obj_type": c.object_type,
                        "rel": c.relation, "timestamp": timestamp_str, "source_type": source_type,
                        "category": c.category, "semantic_type": c.semantic_type,
                        "confidence": c.confidence, "score": c.importance_score,
                        "long_term_useful": c.is_long_term_useful,
                    },
                )

            # "unchanged" means DuplicateDetector issued no write query at
            # all (identical confidence/score already on the edge) -- log
            # that honestly instead of claiming "updated" on a no-op turn.
            # "duplicate" and "new" both involved an actual write (a field
            # refresh, or a new/superseding edge respectively).
            if outcome == "unchanged":
                print(
                    f"[GraphDB unchanged] {c.subject_display} {c.relation} {c.object_display} "
                    f"[{c.semantic_type}, score={c.importance_score:.2f}] -- already up to date"
                )
            else:
                long_term_tag = "long-term" if c.is_long_term_useful else "short-term"
                print(
                    f"[GraphDB updated] ({outcome}) {c.subject_display} {c.relation} {c.object_display} "
                    f"[{c.semantic_type}, score={c.importance_score:.2f}, {long_term_tag}]"
                )

            canonical_facts.append(
                {"subject": c.subject, "relation": c.relation, "object": c.object, "category": c.category}
            )

        return canonical_facts

    # ------------------------------------------------------------------
    # READ PATH
    # ------------------------------------------------------------------
    def extract_query_entities(self, user_query: str) -> list[str]:
        """Lightweight NER over the raw user question, used to seed graph
        expansion. Replaces the previous 'every word >=3 chars' heuristic,
        which pulled in stopwords/verbs as pseudo-entities and produced
        noisy, unscoped traversals.
        """
        prompt = (
            f"Identify the key named entities (people, objects, places, topics) in: '{user_query}'.\n"
            'Return ONLY a JSON array of lowercase strings, e.g. ["keys", "office"]. '
            "No prose, no markdown fences. If none, return []."
        )
        try:
            res = self._llm.chat_once(
                messages=[{"role": "user", "content": prompt}],
                options=self._settings.entity_extraction_options,
            )
            parsed = _parse_json_array(res.cleaned_content)
            if parsed:
                return [str(e).lower().strip() for e in parsed if str(e).strip()]
        except Exception as exc:
            print(f"[Entity Extraction Error] LLM NER failed, falling back to heuristic: {exc}")

        # Fallback: stopword-filtered token split, only if the LLM path failed outright.
        return [w for w in re.findall(r"\b\w{3,}\b", user_query.lower()) if w not in _STOPWORDS]

    def expand_graph_context(self, entities: list[str], categories: list[str] | None = None) -> str:
        """Second Filter: traverses the graph around candidate entities in
        BOTH directions (an entity can be the subject or the object of the
        fact a query is after), optionally scoped to categories carried
        over from the Filter-1 vector hits. Output is grouped by category
        and typed for structured, low-ambiguity LLM consumption.
        """
        if not entities:
            return ""

        rows: list[dict] = []
        seen_edge_keys: set[tuple] = set()
        for entity in entities:
            cypher = (
                "MATCH (s:Entity {name: $entity})-[r:RELATES_TO]-(o:Entity) "
                "RETURN s.name AS sub, s.entity_type AS sub_type, "
                "r.relation AS rel, o.name AS obj, o.entity_type AS obj_type, "
                "r.valid_from AS valid_from, r.valid_to AS valid_to, "
                "r.source_type AS source_type, r.category AS category, r.semantic_type AS semantic_type "
                "ORDER BY r.valid_from DESC LIMIT 10"
            )
            for row in self._db.query_cypher(cypher, {"entity": entity.lower().strip()}):
                if categories and row.get("category") not in categories:
                    continue
                key = (row.get("sub"), row.get("rel"), row.get("obj"), row.get("valid_from"))
                if key in seen_edge_keys:
                    continue
                seen_edge_keys.add(key)
                rows.append(row)

        if not rows:
            return ""

        grouped: dict[str, list[str]] = {}
        for row in rows:
            category = row.get("category") or "uncategorized"
            v_to = row.get("valid_to")
            status = "CURRENT" if not v_to else f"EXPIRED {v_to}"
            semantic_tag = f" ({row['semantic_type']})" if row.get("semantic_type") else ""
            line = (
                f"  [{row.get('sub_type', 'OTHER')}] '{row.get('sub')}' {row.get('rel')} "
                f"[{row.get('obj_type', 'OTHER')}] '{row.get('obj')}'{semantic_tag} "
                f"-- {status}, since {row.get('valid_from', 'unknown')}, via {row.get('source_type', 'text')}"
            )
            grouped.setdefault(category, []).append(line)

        blocks = [f"[{category}]\n" + "\n".join(lines) for category, lines in grouped.items()]
        return "\n".join(blocks)


class EpisodicWorthinessClassifier:
    """Gate that runs BEFORE the Chroma vector-store write, deciding
    whether a turn is worth persisting as episodic memory at all.

    Without this gate, `extract_and_store` wrote every turn to Chroma
    unconditionally (subject only to near-duplicate detection against
    prior entries) -- so greetings ("Hello!"), general-knowledge
    questions ("What is Docker?"), and other turns with zero personal or
    recall value were filling up the vector store right alongside
    genuinely personal statements ("I've practiced calisthenics for
    three years"). A turn that has never been said before isn't a
    duplicate, so the dedup gate alone could never catch this class of
    noise -- it needed its own worthiness check, not a stricter
    duplicate threshold.

    This is treated as a reasoning task for the same reason semantic
    extraction is (see MemoryCandidateExtractor): a keyword/regex list
    can't reliably tell "I am drinking coffee" (personal, arguably worth
    a passing episodic record) apart from "What is Docker?" (textbook
    general knowledge, zero recall value, discard) -- both are short
    declarative-ish sentences with no shared surface pattern to filter
    on. Turns that fail this gate are NOT persisted anywhere: they stay
    in the RAM-only WorkingMemoryBuffer for verbatim continuity on the
    next turn and are logged as "[RAM updated]" by the caller, with no
    Chroma or ArcadeDB write at all.
    """

    def __init__(self, llm_client: LLMClient, settings: Settings):
        self._llm = llm_client
        self._settings = settings

    def is_worth_storing(self, text_content: str) -> bool:
        prompt = (
            "Judge whether this turn is worth storing in long-term episodic memory "
            "(a searchable log of things this user has said or shown, for future recall).\n\n"
            f"Turn: '{text_content}'\n\n"
            "Answer NO if it is a greeting, small talk, a general-knowledge question "
            "(one whose answer doesn't depend on this specific user), a request for "
            "help with something generic, or otherwise has no personal/recall value.\n"
            "Answer YES if it states something personal, an event, a preference, an "
            "observation about the user's situation or surroundings, or anything a "
            "later turn might plausibly need to look back on -- even if it's not "
            "durable enough to become a permanent graph fact.\n\n"
            'Return ONLY a JSON object, no prose, no markdown fences: {"worth_storing": true|false}'
        )
        try:
            res = self._llm.chat_once(
                messages=[{"role": "user", "content": prompt}],
                options=self._settings.entity_extraction_options,
            )
            parsed = _parse_json_object(res.cleaned_content)
            if parsed is not None and "worth_storing" in parsed:
                return bool(parsed["worth_storing"])
        except Exception as exc:
            print(f"[Episodic Worthiness Error] LLM check failed, defaulting to store: {exc}")
        # Fail open: on classifier error, prefer to store rather than
        # silently lose a turn that might have mattered. This is the
        # opposite fail-direction from the semantic pipeline (which
        # fails closed / discards candidates on ambiguity), because
        # losing episodic recall ability is more costly here than one
        # extra stored turn.
        return True




# ==============================================================================
# HYBRID MEMORY MANAGER (Vector Filter 1 + Graph Filter 2)
# ==============================================================================

class CategoryRegistry:
    def __init__(self, chroma_client: chromadb.PersistentClient, settings: Settings):
        self._chroma_client = chroma_client
        self.categories = list(settings.memory_categories)
        # BUGFIX: collections previously used Chroma's default distance
        # space (squared-L2), which is NOT scale-invariant. Vectors here
        # come from gemini-embedding-001, which is not guaranteed to be
        # unit-normalized -- so a raw-L2 distance threshold (e.g. 0.02)
        # was being compared against magnitudes it was never calibrated
        # for, and could silently never fire even for near-identical
        # sentences ("user is drinking coffee" stated twice still logged
        # "[VectorDB updated]" both times instead of "[VectorDB unchanged]"
        # the second time). Forcing hnsw:space="cosine" makes the distance
        # bounded and scale-invariant (0 = identical direction, 2 =
        # opposite), so a fixed threshold means the same thing regardless
        # of embedding magnitude.
        self.collections = {
            cat: chroma_client.get_or_create_collection(
                name=cat, metadata={"hnsw:space": "cosine"}
            )
            for cat in self.categories
        }

    def total_entry_count(self) -> int:
        return sum(col.count() for col in self.collections.values())


class HybridMemoryManager:
    """Hybrid RAG implementation combining Dense Vector Search and Temporal Graphiti expansion."""

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        registry: CategoryRegistry,
        arcadedb: ArcadeDBClient,
        settings: Settings,
    ):
        self._llm = llm_client
        self._embedder = embedding_client
        self._registry = registry
        self._settings = settings
        self._graph_engine = GraphitiTemporalEngine(arcadedb, llm_client, settings)
        self._worthiness = EpisodicWorthinessClassifier(llm_client, settings)

    def query(self, user_query: str) -> str:
        """Two-Stage Hybrid RAG Pipeline.

        Filter 1 (vector similarity) and Filter 2 (graph expansion) are
        linked structurally rather than by regex guesswork: each vector
        hit carries the exact entity names extracted for it at write time
        (see `extract_and_store`), plus the category it was written under.
        Those names seed the graph traversal, and that same category
        scopes it, so Filter 2 expands the actual neighborhood of what
        Filter 1 found instead of an unrelated word-overlap match.
        """
        # --- FILTER 1: Vector Similarity Search ---
        query_vector = self._embedder.embed(user_query, task_type="RETRIEVAL_QUERY")
        if not query_vector:
            return ""

        vector_results = []
        seed_entities: set[str] = set()
        hit_categories: set[str] = set()

        for category, col in self._registry.collections.items():
            if col.count() == 0:
                continue
            res = col.query(query_embeddings=[query_vector], n_results=2, include=["documents", "metadatas"])
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            for doc, meta in zip(docs, metas):
                vector_results.append(f"[{category}] {doc}")
                hit_categories.add(category)
                for name in (meta or {}).get("entities", "").split(","):
                    name = name.strip()
                    if name:
                        seed_entities.add(name)

        vector_context_block = "\n".join(vector_results)

        # --- FILTER 2: Temporal Graph Expansion (Graphiti + ArcadeDB) ---
        # Always include entities named directly in the query itself
        # (e.g. "where are my keys" -> "keys" even if no vector hit
        # mentioned it yet), on top of the entities inherited from hits.
        seed_entities.update(self._graph_engine.extract_query_entities(user_query))
        graph_context_block = self._graph_engine.expand_graph_context(
            list(seed_entities), categories=list(hit_categories) or None
        )

        # --- COMBINE FILTER RESULTS ---
        hybrid_context = []
        if vector_context_block:
            hybrid_context.append("=== Episodic Memory Matches (Filter 1 — vector similarity) ===\n" + vector_context_block)
        if graph_context_block:
            hybrid_context.append("=== Semantic Memory Connections (Filter 2 — temporal graph expansion) ===\n" + graph_context_block)

        return "\n\n".join(hybrid_context)

    def extract_and_store(self, user_input: str, visual_context: str | None = None) -> bool:
        """Runs the semantic memory pipeline against ArcadeDB first, then
        -- gated by episodic worthiness, not written unconditionally --
        writes the raw turn to the Chroma Vector Store (episodic memory),
        tagging it with whatever entity names/category the graph step
        produced (possibly none -- most turns yield no durable fact, and
        that's the graph staying a semantic store instead of a
        transcript). Entities are extracted once, so both stores end up
        referencing the same names and read-time retrieval never has to
        re-derive them heuristically.

        Returns True if this turn was persisted anywhere (GraphDB and/or
        VectorDB), False if it stayed RAM-only (caller should log
        "[RAM updated]" in that case, since nothing here was written to
        disk).
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")
        category = "visual_memories" if visual_context else "conversations"
        source_type = "image" if visual_context else "text"
        text_to_store = f"{user_input} | Visual: {visual_context}" if visual_context else user_input

        # 1. Extract and Update Temporal Knowledge Graph in ArcadeDB.
        # This stage already has its own internal gate (semantic-type fit
        # + importance threshold), so a greeting or general-knowledge
        # question legitimately produces `facts == []` here without any
        # extra check needed at this call site.
        facts = self._graph_engine.extract_and_upsert_facts(
            text_to_store, timestamp_str=now_str, default_category=category, source_type=source_type
        )
        persisted_anywhere = bool(facts)
        entity_names = sorted({f["subject"] for f in facts} | {f["object"] for f in facts})
        # A fact's own category (as classified by the extraction step) may
        # differ from the collection this text is filed under (e.g. a
        # profile-level preference mentioned during casual conversation);
        # store the file-category for retrieval but keep it consistent.
        resolved_category = facts[0]["category"] if facts else category

        # 2. Episodic worthiness gate -- BUGFIX: previously the Chroma
        # write below ran unconditionally for every turn (subject only
        # to near-duplicate detection against prior entries), so turns
        # with zero personal/recall value -- "Hello!", "What is Docker?"
        # -- were filling episodic memory just like genuinely personal
        # statements. A turn that's never been said before isn't a
        # duplicate, so the dedup gate alone could never catch this; it
        # needed its own check. Skip straight to the RAM-only outcome if
        # this turn doesn't clear it and the graph step above found
        # nothing either.
        if not facts and not self._worthiness.is_worth_storing(text_content=text_to_store):
            return persisted_anywhere  # False: nothing written to either store

        # 3. Store in Chroma Vector Store, carrying the linked entity names --
        # but only if this text isn't a near-duplicate of something already
        # sitting in the same collection. Without this gate, every turn
        # (including "hello" said twice, or a question re-asked verbatim)
        # was written unconditionally, so episodic memory just grew as a
        # transcript rather than a de-duplicated log of distinct turns.
        vector = self._embedder.embed(text_to_store)
        if not vector:
            return persisted_anywhere

        collection = self._registry.collections[category]
        is_duplicate = False
        if collection.count() > 0:
            existing = collection.query(
                query_embeddings=[vector], n_results=1, include=["documents", "distances"]
            )
            existing_docs = existing.get("documents", [[]])[0]
            existing_dists = existing.get("distances", [[]])[0]
            if existing_docs and existing_dists:
                # Collections are created with hnsw:space="cosine" (see
                # CategoryRegistry), so this is a bounded, scale-invariant
                # distance (0 = identical direction, 2 = opposite) --
                # not raw L2, which would be skewed by the magnitude of
                # whatever embedding model produced the vector.
                if existing_dists[0] <= self._settings.episodic_duplicate_distance_threshold:
                    is_duplicate = True

        if is_duplicate:
            print(f"[VectorDB unchanged] [{category}] near-duplicate skipped: {text_to_store[:80]!r}")
            return persisted_anywhere

        collection.add(
            embeddings=[vector],
            documents=[text_to_store],
            metadatas=[{
                "timestamp": time.time(),
                "formatted_time": now_str,
                "entities": ",".join(entity_names),
                "graph_category": resolved_category,
            }],
            ids=[str(uuid.uuid4())],
        )
        print(f"[VectorDB updated] [{category}] {text_to_store[:80]!r}")
        return True

    def total_entry_count(self) -> int:
        return self._registry.total_entry_count()


class WorkingMemoryBuffer:
    """Layer 1 of the three-tier memory architecture: a small ring buffer
    of verbatim recent turns (+ a pointer to any camera context attached
    to those turns), held in-process only. Never written to Chroma or
    ArcadeDB -- there is deliberately no '[...updated]' log line for this
    layer, since it has no store to update.

    The problem this solves: without it, every turn's `messages` list
    contained only the system prompt and the current question, so the
    model had zero verbatim continuity with what was just said. Pronouns
    ("it", "that"), quick follow-ups ("what about the second one?"), and
    short back-and-forth exchanges had nothing to resolve against except
    whatever the episodic/semantic retrieval happened to surface.

    This is intentionally NOT a substitute for the other two layers --
    it holds only the last `max_turns` exchanges, unscored and
    unfiltered, and is gone the moment the process exits:
      - Working memory  (this class): "what did we just say?"
        verbatim, seconds-to-minutes, RAM only, never persisted.
      - Episodic memory (Chroma):     "have we discussed something
        like this before?" similarity search, persisted to disk.
      - Semantic memory (ArcadeDB):   "what do I know is durably
        true?" scored/filtered facts, persisted to the graph.

    Hard boundary
    -------------
    This class has NO reference to a Chroma client, an ArcadeDB client,
    an embedder, or an LLM client -- by construction, not just by
    convention. It cannot write to either downstream store even by
    accident, because it holds no handle to either. `has_persistence_backend`
    is a permanent `False` sentinel so a caller (or a future refactor)
    can assert this boundary at runtime instead of relying on someone
    reading this docstring; `assert_no_persistence()` is a cheap guard
    other components can call defensively.
    """

    has_persistence_backend: bool = False

    def __init__(self, max_turns: int = 5):
        self._max_turns = max(0, max_turns)
        self._turns: deque[dict[str, Any]] = deque(maxlen=self._max_turns or None)

    def assert_no_persistence(self) -> None:
        """Defensive boundary check: raises if this instance somehow
        acquired a persistence handle (e.g. someone later adds a
        `self._db` or `self._chroma` attribute to this class without
        updating this guard). Cheap enough to call from a health check
        or from tests; not on the hot path."""
        forbidden_attrs = ("_db", "_chroma", "_arcadedb", "_embedder", "_llm", "_registry")
        leaked = [a for a in forbidden_attrs if hasattr(self, a)]
        if leaked or self.has_persistence_backend:
            raise RuntimeError(
                f"WorkingMemoryBuffer boundary violation: found persistence-like "
                f"attribute(s) {leaked} -- working memory must stay RAM-only and "
                f"must never hold a Graph/Vector DB handle."
            )

    def as_messages(self) -> list[dict]:
        """Buffered turns as alternating user/assistant messages, oldest
        first -- splice these between the system prompt and the current
        question in the chat `messages` list. Camera/visual context that
        was attached to a buffered turn is folded into the stored user
        text at `add_turn` time (see `camera_context` there) so it rides
        along here as plain conversational text -- it is never handed
        back out as a separate image/tool payload, and it never touches
        Chroma or ArcadeDB from this class."""
        messages: list[dict] = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        return messages

    def add_turn(
        self,
        user_question: str,
        assistant_response: str,
        camera_context: str | None = None,
    ) -> None:
        """Record one turn. `camera_context` (e.g. 'Frame captured' or a
        short description of what the camera saw this turn) is optional
        and, if present, is folded into the stored user text as a plain
        annotation -- it stays scoped to this RAM buffer exactly like the
        rest of the turn, with no separate path to any persisted store."""
        if self._max_turns <= 0:
            return
        stored_user_text = (
            f"{user_question} [camera: {camera_context}]" if camera_context else user_question
        )
        self._turns.append({"user": stored_user_text, "assistant": assistant_response})


# ==============================================================================
# ROUTER & AGENT COMPONENTS
# ==============================================================================

class Route(enum.Enum):
    SIMPLE = "simple"
    PLAN = "plan"


class RouteComplexity(enum.Enum):
    """Sub-tier of Route.PLAN. The old binary router gave a single
    web lookup ('what's the weather') the same treatment as a multi-step
    comparative task ('find the cheapest flight and check if it clashes
    with my calendar') — either capping the former with unneeded planning
    overhead, or capping the latter's round budget too early to finish.
    SINGLE_STEP skips the planning/critique steps and runs a tight
    1-round tool call. MULTI_STEP gets the full plan -> execute -> reflect
    loop with its own round budget.
    """
    NONE = "none"           # Route.SIMPLE — no tools involved
    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"


class ToolDomain(enum.Enum):
    """Coarse groupings used to scope tool schemas per PLAN turn. Prevents
    every registered tool being passed into every tool-orchestration call
    as the tool count grows past a handful."""
    WEB = "web"
    PRODUCTIVITY = "productivity"
    COMMUNICATION = "communication"
    NAVIGATION = "navigation"
    MEMORY = "memory"
    BROWSER = "browser"


@dataclass
class RoutingDecision:
    route: Route
    reason: str


@dataclass
class ExtendedRoutingDecision:
    route: Route
    reason: str
    complexity: RouteComplexity = RouteComplexity.NONE
    target_domains: list[ToolDomain] = field(default_factory=list)


class Router:
    """Classifies each turn as SIMPLE (answerable directly, from memory,
    general knowledge, or injected situational context) or PLAN (requires
    tool use: web search/fetch, multi-step research, comparison shopping,
    checking live prices/schedules/availability).

    The previous version fell through to a hardcoded 'default simple' for
    anything that didn't match a five-word keyword list. That silently
    misrouted any complex agentic request phrased without those exact
    words (e.g. 'book the cheapest flight from Venice to Bilbao') into
    the no-tools path, where the model could only apologize and promise
    to check later. Fast, cheap heuristics are kept for the unambiguous
    cases; everything else goes through an actual classification call
    instead of guessing.
    """

    _FAST_PLAN_KEYWORDS = (
        "search", "news", "fetch", "weather", "latest",
        "book", "flight", "flights", "price", "prices", "buy", "purchase",
        "ticket", "tickets", "schedule", "availability", "cheapest", "fastest",
        "quickest", "reserve", "reservation", "score", "stock", "exchange rate",
    )
    _FAST_SIMPLE_PATTERN = re.compile(
        r"\bwhat (time|day|date) is it\b|\bwhere am i\b|\bwhat'?s the time\b", re.IGNORECASE
    )

    def __init__(self, llm_client: LLMClient, settings: Settings):
        self._llm = llm_client
        self._settings = settings

    def classify(self, user_question: str) -> RoutingDecision:
        lowered = user_question.lower()

        # Fast path 1: directly answerable from injected situational context
        # (current time/date/location) — no tool call needed, no LLM call needed.
        if self._FAST_SIMPLE_PATTERN.search(lowered):
            return RoutingDecision(Route.SIMPLE, "direct situational-context query (time/date/location)")

        # Fast path 2: unambiguous planning keywords — skip the round trip.
        if any(kw in lowered for kw in self._FAST_PLAN_KEYWORDS):
            return RoutingDecision(Route.PLAN, "planning keyword matched")

        # Fast path 3: very short queries are almost always greetings/chit-chat.
        if len(lowered.split()) <= 3:
            return RoutingDecision(Route.SIMPLE, "short greeting/query")

        # Everything else: ask the model. This is the fix — no more silent
        # 'default simple' for requests the keyword list doesn't anticipate.
        return self._classify_via_llm(user_question)

    def _classify_via_llm(self, user_question: str) -> RoutingDecision:
        prompt = (
            "Classify the user request below as exactly one word: PLAN or SIMPLE.\n"
            "PLAN: requires live or external information, multi-step research, comparison shopping, "
            "checking current prices, schedules, availability, or booking-type requests, or anything that "
            "needs a tool call (web search or web fetch) before it can be answered accurately.\n"
            "SIMPLE: can be answered directly from memory, general knowledge, opinion, or conversation, "
            "with no need for current external data.\n"
            f"Request: '{user_question}'\n"
            "Answer with exactly one word: PLAN or SIMPLE."
        )
        try:
            res = self._llm.chat_once(
                messages=[{"role": "user", "content": prompt}],
                options=self._settings.router_options,
            )
            cleaned = res.cleaned_content.strip().upper()
        except Exception as exc:
            print(f"[Thinking] Router LLM classification failed ({exc}); defaulting to PLAN.")
            return RoutingDecision(Route.PLAN, "classifier call failed, defaulting to plan")

        if "PLAN" in cleaned:
            return RoutingDecision(Route.PLAN, "llm classifier: plan")
        if "SIMPLE" in cleaned:
            return RoutingDecision(Route.SIMPLE, "llm classifier: simple")
        # Malformed/ambiguous output: fail toward capability (run the agent
        # loop with tools available) rather than toward a silent, tool-less
        # refusal — the failure mode this fix targets.
        return RoutingDecision(Route.PLAN, f"llm classifier ambiguous ({cleaned!r}), defaulting to plan")


class AdvancedRouter(Router):
    """Resolves route (SIMPLE/PLAN), complexity (single_step/multi_step) and
    tool domains together.

    v1.8 changes, replacing the two-call, budget-starved, keyword-gated
    design:

    1. ONE merged, reasoning-enabled LLM call instead of two sequential
       round trips (macro classify() then a separate complexity/domain
       call). The previous design spent two API calls, each with a
       5-token budget, to make one decision — no room for the model to
       reason before answering. The merged call asks for a short rationale
       *before* the label and gets ~200 tokens of headroom
       (router_reasoning_options), which for a mid-size model reliably
       beats zero-shot label-only classification, at roughly the same or
       lower total token cost than the old two-call design.
    2. Self-consistency voting for the genuinely ambiguous middle: queries
       that don't hit a fast-path fire the merged classifier N times
       (router_self_consistency_samples) at slightly jittered temperature;
       route/complexity are decided by majority vote instead of trusting a
       single deterministic sample. This buys calibration without a bigger
       model — it's redundancy, not capability.
    3. Fast-path heuristics kept, but tightened. The previous version
       fired MULTI_STEP on a bare ' vs ' or PLAN on the bare word 'score' —
       both common in phrasings that don't need any tool call at all
       ('Python vs JavaScript, which should I learn?', 'score this essay').
       A wrong fast-path is worse than a slightly slower correct
       classification, because nothing downstream ever re-checks a fast
       path once it fires. Fast paths now require a comparison marker
       *and* an execution verb (book/buy/reserve/schedule) for MULTI_STEP,
       and drop ambiguous single words like 'score' entirely — those now
       go through the reasoning-enabled classifier instead of a guess.
    4. Fast-path hits and self-consistency vote splits are logged
       (console + optional file), so heuristic tuning is evidence-based
       instead of guesswork.
    """

    _DOMAIN_CHOICES = ", ".join(d.value for d in ToolDomain)

    _FAST_SIMPLE_CHITCHAT_PATTERN = re.compile(
        r"\bwhat (time|day|date) is it\b|\bwhere am i\b|\bwhat'?s the time\b", re.IGNORECASE
    )

    # Kept deliberately narrow: only phrasings that are unambiguous even
    # with a "who is/what is" prefix stripped away, and where the target
    # concept is clearly a live/external lookup, not a matter of opinion.
    _FAST_SINGLE_STEP_PATTERN = re.compile(
        r"\bweather\b|\bstock price\b|\bexchange rate\b|^\s*(what'?s|what is) the (time|weather) in\b",
        re.IGNORECASE,
    )
    _FAST_PLAN_KEYWORDS = ("flight", "flights", "book a", "buy tickets", "purchase tickets")

    # A comparison marker alone ('vs', 'compare') is not enough — that's as
    # likely to be an opinion question as a task. Multi-step only fires
    # when a comparison co-occurs with an execution verb implying the
    # assistant should actually go do something with the result.
    _COMPARISON_MARKERS = re.compile(r"\bcompare\b|\bvs\.?\b|\bversus\b", re.IGNORECASE)
    _EXECUTION_VERBS = re.compile(
        r"\bbook\b|\bbuy\b|\breserve\b|\bschedule\b|\bplan (my|a|an)\b|\bfind (me )?the cheapest\b",
        re.IGNORECASE,
    )
    _FAST_MULTI_STEP_KEYWORDS = ("and then", "after that", "itinerary", "cheapest and", "book and")

    # Fast path: explicit browser-interaction verbs that need a live session
    # (clicking, filling forms, navigating through multi-page flows) — the
    # model should route to the browse_website tool rather than web_search.
    _FAST_BROWSER_INTERACTION_PATTERN = re.compile(
        r"\b(buy|purchase|book|reserve|order|fill (out|in)|sign up|register|subscribe|"
        r"add to (cart|basket)|check(?:out| out)|pay|login|log in|sign in)\b",
        re.IGNORECASE,
    )

    def classify_advanced(self, user_question: str) -> ExtendedRoutingDecision:
        lowered = user_question.lower()

        # Fast path 1: directly answerable from injected situational
        # context, or trivially short chit-chat — no tool call, no LLM call.
        if self._FAST_SIMPLE_CHITCHAT_PATTERN.search(lowered):
            return self._fast_result(user_question, Route.SIMPLE, "direct situational-context query", RouteComplexity.NONE)
        if len(lowered.split()) <= 3:
            return self._fast_result(user_question, Route.SIMPLE, "short greeting/query", RouteComplexity.NONE)

        # Fast path 2: unambiguous single-tool-call lookups.
        if any(kw in lowered for kw in self._FAST_PLAN_KEYWORDS) or self._FAST_SINGLE_STEP_PATTERN.search(lowered):
            return self._fast_result(
                user_question, Route.PLAN, "planning keyword/pattern matched", RouteComplexity.SINGLE_STEP,
                [ToolDomain.WEB, ToolDomain.MEMORY],
            )

        # Fast path 3: comparison + execution intent, or explicit sequencing
        # language — genuinely needs multiple tool calls, not just an
        # opinion between two options.
        needs_execution_comparison = bool(self._COMPARISON_MARKERS.search(lowered) and self._EXECUTION_VERBS.search(lowered))
        if needs_execution_comparison or any(kw in lowered for kw in self._FAST_MULTI_STEP_KEYWORDS):
            return self._fast_result(
                user_question, Route.PLAN, "multi-step comparison/sequencing pattern matched", RouteComplexity.MULTI_STEP,
                [ToolDomain.WEB, ToolDomain.MEMORY],
            )

        # Fast path 4: explicit browser-interaction verbs — needs a live
        # session with click/type/navigate, not just a text fetch.
        if self._FAST_BROWSER_INTERACTION_PATTERN.search(lowered):
            return self._fast_result(
                user_question, Route.PLAN, "browser interaction verb matched", RouteComplexity.MULTI_STEP,
                [ToolDomain.BROWSER, ToolDomain.WEB, ToolDomain.MEMORY],
            )

        # Everything else: no more silent 'default simple' and no more
        # single-shot, budget-starved guessing — a reasoning-enabled merged
        # call, voted over several samples for the ambiguous middle.
        return self._classify_via_llm_merged(user_question)

    # ------------------------------------------------------------------
    # Merged, reasoning-enabled classification with self-consistency
    # ------------------------------------------------------------------
    def _classify_via_llm_merged(self, user_question: str) -> ExtendedRoutingDecision:
        n = max(1, self._settings.router_self_consistency_samples)
        jitter = self._settings.router_self_consistency_temp_jitter
        votes: list[dict] = []
        for i in range(n):
            sample = self._single_merged_call(user_question, temperature_offset=i * jitter)
            if sample is not None:
                votes.append(sample)

        if not votes:
            # Fail toward capability: a confidently-wrong SIMPLE answer with
            # no tool available is the worse failure mode than a couple of
            # extra rounds spent on a task that turns out to be simpler
            # than expected.
            print("[Thinking] Merged router classification failed on all samples; defaulting to PLAN/MULTI_STEP.")
            return ExtendedRoutingDecision(
                Route.PLAN, "all classifier samples failed, defaulting to plan/multi_step",
                RouteComplexity.MULTI_STEP, [ToolDomain.WEB, ToolDomain.MEMORY],
            )

        route_tally = Counter(v["route"] for v in votes)
        winning_route, route_count = route_tally.most_common(1)[0]
        if route_count < len(votes):
            self._log_disagreement(user_question, votes)

        if winning_route == Route.SIMPLE:
            reason = f"llm classifier majority ({route_count}/{len(votes)}): simple"
            return ExtendedRoutingDecision(Route.SIMPLE, reason, RouteComplexity.NONE)

        plan_votes = [v for v in votes if v["route"] == Route.PLAN]
        complexity_tally = Counter(v["complexity"] for v in plan_votes)
        winning_complexity = complexity_tally.most_common(1)[0][0] if complexity_tally else RouteComplexity.MULTI_STEP

        domains: set[ToolDomain] = set()
        for v in plan_votes:
            domains.update(v["domains"])
        if not domains:
            domains = {ToolDomain.WEB}
        domains.add(ToolDomain.MEMORY)

        reason = f"llm classifier majority ({route_count}/{len(votes)}): plan/{winning_complexity.value}"
        return ExtendedRoutingDecision(
            Route.PLAN, reason, winning_complexity, sorted(domains, key=lambda d: d.value)
        )

    def _single_merged_call(self, user_question: str, temperature_offset: float = 0.0) -> dict | None:
        prompt = (
            "You are a routing classifier for an AI assistant with tool access. Reason briefly, then decide.\n"
            f"Request: '{user_question}'\n\n"
            "Step 1 (reasoning): in one short sentence, note what the request actually needs — an opinion/"
            "memory answer, a single live lookup, or multiple tool calls with comparison or sequencing.\n"
            "Step 2 (route): exactly one of SIMPLE (answerable from memory, general knowledge, opinion, or "
            "conversation — no current external data needed) or PLAN (needs live/external info or a tool call).\n"
            "Step 3 (complexity, only if route=PLAN): none, single_step (one tool call, no further reasoning "
            "over the result), or multi_step (multiple tool calls, comparison, sequencing, research, booking).\n"
            f"Step 4 (domains, only if route=PLAN): minimum required tool domains from [{self._DOMAIN_CHOICES}].\n\n"
            "Return your reasoning, then end your response with exactly one JSON object on its own line and "
            "nothing after it:\n"
            '{"reasoning": "...", "route": "SIMPLE"|"PLAN", "complexity": "none"|"single_step"|"multi_step", '
            '"domains": ["..."]}'
        )
        options = SamplingOptions(
            temperature=min(1.0, self._settings.router_reasoning_options.temperature + temperature_offset),
            top_p=self._settings.router_reasoning_options.top_p,
            num_predict=self._settings.router_reasoning_options.num_predict,
        )
        try:
            res = self._llm.chat_once(messages=[{"role": "user", "content": prompt}], options=options)
            cleaned = res.cleaned_content.strip()
            # The label is the LAST JSON object in the output — reasoning
            # text is expected to precede it, so parse off the tail rather
            # than assuming the whole response is JSON.
            match = re.findall(r"\{.*?\}", cleaned, re.DOTALL)
            parsed = json.loads(match[-1]) if match else json.loads(cleaned)
        except Exception as exc:
            print(f"[Thinking] Merged router sample failed ({exc}); skipping this vote.")
            return None

        route_raw = str(parsed.get("route", "")).upper()
        if route_raw not in ("SIMPLE", "PLAN"):
            return None
        route = Route.PLAN if route_raw == "PLAN" else Route.SIMPLE

        complexity_raw = str(parsed.get("complexity", "")).lower()
        if complexity_raw == "multi_step":
            complexity = RouteComplexity.MULTI_STEP
        elif complexity_raw == "single_step":
            complexity = RouteComplexity.SINGLE_STEP
        else:
            complexity = RouteComplexity.NONE

        domains = [d for d in ToolDomain if d.value in [str(x).lower() for x in parsed.get("domains", [])]]
        return {"route": route, "complexity": complexity, "domains": domains}

    # ------------------------------------------------------------------
    # Logging (evidence for future heuristic tuning, not just guesswork)
    # ------------------------------------------------------------------
    def _fast_result(
        self,
        user_question: str,
        route: Route,
        reason: str,
        complexity: RouteComplexity,
        domains: list[ToolDomain] | None = None,
    ) -> ExtendedRoutingDecision:
        self._append_routing_log(f"FAST_PATH\troute={route.value}\treason={reason}\tquery={user_question!r}")
        return ExtendedRoutingDecision(route, reason, complexity, domains or [])

    def _log_disagreement(self, user_question: str, votes: list[dict]) -> None:
        vote_summary = [f"{v['route'].value}/{v['complexity'].value}" for v in votes]
        print(f"[Thinking] Self-consistency votes split {vote_summary} for: {user_question!r}")
        self._append_routing_log(f"VOTE_SPLIT\tvotes={vote_summary}\tquery={user_question!r}")

    def _append_routing_log(self, line: str) -> None:
        path = self._settings.routing_log_path
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


class ScalableToolRegistry:
    """Domain-scoped replacement for the flat ToolRegistry. Schemas are
    grouped by ToolDomain so a PLAN turn can request only the domains the
    AdvancedRouter selected, keeping tool-call context bounded as the
    number of registered tools grows."""

    def __init__(self):
        self._domains: dict[ToolDomain, list[Any]] = {domain: [] for domain in ToolDomain}
        self._implementations: dict[str, Callable[..., Any]] = {}

    def register(self, domain: ToolDomain, name: str, schema: Any, implementation: Callable[..., Any]) -> None:
        self._domains[domain].append(schema)
        self._implementations[name] = implementation

    def schemas_for(self, domains: Sequence[ToolDomain]) -> list[Any]:
        schemas: list[Any] = []
        for domain in domains:
            schemas.extend(self._domains.get(domain, []))
        return schemas

    def all_schemas(self) -> list[Any]:
        return self.schemas_for(list(self._domains.keys()))

    def get_implementation(self, name: str) -> Callable[..., Any] | None:
        return self._implementations.get(name)


class ToolExecutor:
    def __init__(self, registry: ScalableToolRegistry):
        self._registry = registry

    def execute(self, tool_call: Any) -> str:
        fn_name = tool_call.function.name
        impl = self._registry.get_implementation(fn_name)
        if not impl:
            return f"TOOL_ERROR: unknown tool {fn_name}"
        try:
            res = impl(**tool_call.function.arguments)
            return str(res)
        except Exception as exc:
            return f"TOOL_ERROR: {exc}"


class PlanExecuteReflectAgent:
    """Replaces the flat, single-pattern bounded-ReAct loop with a
    complexity-tiered engine:

    - RouteComplexity.SINGLE_STEP: tight 1-round tool call, no planning
      overhead. v1.8 adds a cheap post-hoc check even here (see below) —
      the previous version had no safety net at all on this path: a
      confidently-wrong SINGLE_STEP classification just returned whatever
      the one tool round produced, with no recovery.
    - RouteComplexity.MULTI_STEP: explicit Plan -> Execute -> Reflect:
        1. Plan: deconstruct the query into an ordered sub-goal list
           before any tool call is made.
        2. Execute: domain-scoped tool calls, guarded against repeating
           an identical failed call, with a reflection note on error.
        3. Reflect: a post-loop critique call checks whether the
           gathered tool output actually answers the query; if not,
           and budget remains, it feeds a corrective instruction back
           for one more execution pass instead of silently returning
           an incomplete answer.

    v1.8 escape hatch: a SINGLE_STEP run now also gets the critique check
    applied to it. If the tool result doesn't actually look like it
    answers the query — the same failure the reflect loop already guards
    against for MULTI_STEP — the turn escalates into the full
    plan->execute->reflect loop instead of returning a broken answer with
    no recovery path. This reuses the existing _critique machinery rather
    than adding a second, parallel verification mechanism.
    """

    def __init__(self, llm_client: LLMClient, registry: ScalableToolRegistry, executor: ToolExecutor, settings: Settings):
        self._llm = llm_client
        self._registry = registry
        self._executor = executor
        self._settings = settings

    def _pre_execution_analysis(self, messages: list, on_thinking: Callable | None = None) -> str | None:
        """Analyzes whether the agent has enough information to proceed with
        tool use.  Returns a clarifying question string if critical
        information is missing, or None if the agent should proceed.

        This runs a cheap LLM call that inspects the user's request
        against the available context (system prompt carrying memory,
        location, time) and asks itself: 'What do I still need to know
        before I can use my tools effectively?'  If the answer is
        something only the user can provide (budget, food preference,
        exact dates, etc.), the agent pauses and asks rather than
        guessing and burning tool rounds on a wrong assumption.
        """
        if on_thinking:
            on_thinking(-1)

        user_question = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if not user_question or len(user_question.split()) <= 2:
            return None

        system_context = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        context_excerpt = system_context[:3000] if system_context else "(no context available)"

        prompt = (
            f"{CLARIFICATION_ANALYSIS_PROMPT}\n\n"
            f"User's request: '{user_question}'\n\n"
            f"Available context:\n{context_excerpt}"
        )

        try:
            res = self._llm.chat_once(
                messages=[{"role": "user", "content": prompt}],
                options=self._settings.planning_options,
            )
            cleaned = res.cleaned_content.strip()
            match = re.findall(r"\{.*?\}", cleaned, re.DOTALL)
            parsed = json.loads(match[-1]) if match else json.loads(cleaned)
        except Exception as exc:
            print(f"[Thinking] Clarification analysis failed ({exc}); proceeding with tools.")
            return None

        if parsed.get("proceed", True):
            return None

        reasoning = str(parsed.get("reasoning", "")).strip()
        question = str(parsed.get("question", "")).strip()
        if reasoning:
            print(f"[Thinking] Clarification needed: {reasoning}")
        return question if question else None

    def run(
        self,
        messages: list,
        domains: list[ToolDomain] | None = None,
        complexity: RouteComplexity = RouteComplexity.MULTI_STEP,
        on_tool_call=None,
        on_thinking=None,
        on_plan=None,
        on_reflect=None,
        on_clarification=None,
    ) -> list:
        active_schemas = self._registry.schemas_for(domains) if domains else self._registry.all_schemas()

        # Pre-execution clarification check: before burning tool rounds on
        # a potentially under-specified request, ask the LLM whether any
        # critical information is missing that only the user can provide.
        # This catches cases like "find me a restaurant" where location,
        # budget, or food preference should be clarified first rather than
        # assumed.  Only fires when tools will actually be used (PLAN route).
        if complexity != RouteComplexity.NONE:
            question = self._pre_execution_analysis(messages, on_thinking)
            if question:
                if on_clarification:
                    on_clarification(question)
                messages.append({
                    "role": "system",
                    "content": (
                        "[SYSTEM NOTE: You need more information before proceeding with this task. "
                        "Ask the user this question naturally in your response, and wait for their "
                        "answer before taking any action. Do NOT use any tools yet.]\n\n"
                        f"Question to ask: {question}"
                    ),
                })
                return messages

        if complexity == RouteComplexity.SINGLE_STEP:
            self._execute_rounds(
                messages, active_schemas, self._settings.single_step_max_rounds, on_tool_call, on_thinking
            )

            if not self._settings.single_step_escalate_to_multi_step:
                return messages

            # Post-hoc verifier: did the one tool round actually answer the
            # request? Reuses the same critique call MULTI_STEP already
            # pays for — SINGLE_STEP just wasn't reachable from it before.
            verdict, gap = self._critique(messages, on_reflect)
            if verdict != "insufficient":
                return messages

            messages.append({
                "role": "system",
                "content": f"[SYSTEM REFLECTION NOTE: Single-step result was insufficient — {gap} "
                            "Escalating to full multi-step planning and tool use.",
            })
            return self._run_multi_step(messages, active_schemas, on_tool_call, on_thinking, on_plan, on_reflect)

        return self._run_multi_step(messages, active_schemas, on_tool_call, on_thinking, on_plan, on_reflect)

    def _run_multi_step(
        self,
        messages: list,
        active_schemas: list,
        on_tool_call,
        on_thinking,
        on_plan,
        on_reflect,
    ) -> list:
        # MULTI_STEP: explicit planning step before any tool call.
        plan = self._deconstruct_plan(messages, on_plan)
        if plan:
            messages.append({
                "role": "system",
                "content": "Sub-goals for this request, address them in order:\n" + "\n".join(f"- {g}" for g in plan),
            })

        for attempt in range(self._settings.max_reflection_retries + 1):
            self._execute_rounds(
                messages, active_schemas, self._settings.max_agent_tool_rounds, on_tool_call, on_thinking
            )

            verdict, critique = self._critique(messages, on_reflect)
            if verdict != "insufficient":
                break
            if attempt < self._settings.max_reflection_retries:
                messages.append({
                    "role": "system",
                    "content": f"[SYSTEM REFLECTION NOTE: Gathered information is insufficient — {critique} "
                                "Continue with additional or different tool calls to close the gap.",
                })

        return messages

    # ------------------------------------------------------------------
    # PLAN
    # ------------------------------------------------------------------
    def _deconstruct_plan(self, messages: list, on_plan) -> list[str]:
        user_question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        prompt = (
            f"Deconstruct this task into an ordered list of concrete sub-goals: '{user_question}'\n"
            'Return ONLY a JSON array of short strings, e.g. ["find X", "compare X and Y"]. Max 5 items.'
        )
        try:
            res = self._llm.chat_once(
                messages=[{"role": "user", "content": prompt}],
                options=self._settings.planning_options,
            )
            raw = re.sub(r"^```(?:json)?|```$", "", res.cleaned_content.strip(), flags=re.MULTILINE).strip()
            plan = [str(g) for g in json.loads(raw)][:5]
        except Exception as exc:
            print(f"[Thinking] Plan deconstruction failed ({exc}); proceeding without an explicit plan.")
            plan = []
        if on_plan:
            on_plan(plan)
        return plan

    # ------------------------------------------------------------------
    # EXECUTE
    # ------------------------------------------------------------------
    def _execute_rounds(self, messages: list, active_schemas: list, max_rounds: int, on_tool_call, on_thinking) -> None:
        seen_calls: set[tuple] = set()

        for round_num in range(1, max_rounds + 1):
            if on_thinking:
                on_thinking(round_num)
            res = self._llm.chat_once(
                messages=messages,
                tools=active_schemas,
                options=self._settings.tool_orchestration_options,
            )
            raw = res.raw_message
            messages.append(raw)
            if not raw.tool_calls:
                return

            for call in raw.tool_calls:
                if on_tool_call:
                    on_tool_call(call.function.name, call.function.arguments)

                call_key = (call.function.name, tuple(sorted(call.function.arguments.items())))
                if call_key in seen_calls:
                    note = (
                        f"Tool '{call.function.name}' was already called with these exact arguments. "
                        "Repeating it will not produce a different result — change the arguments, "
                        "try a different tool, or stop and answer with what you have."
                    )
                    messages.append({
                        "role": "tool",
                        "content": f"[SYSTEM REFLECTION NOTE: {note}]",
                        "tool_name": call.function.name,
                    })
                    continue
                seen_calls.add(call_key)

                out = self._executor.execute(call)
                if out.startswith("TOOL_ERROR") or "Error" in out[:32]:
                    note = (
                        f"Tool '{call.function.name}' failed with output: {out}.\n"
                        "Analyze why it failed and adjust your approach. Do NOT repeat the exact same call."
                    )
                    content = f"{out}\n\n[SYSTEM REFLECTION NOTE: {note}]"
                else:
                    content = out[:4000]
                messages.append({"role": "tool", "content": content, "tool_name": call.function.name})

    # ------------------------------------------------------------------
    # REFLECT / CRITIQUE
    # ------------------------------------------------------------------
    def _critique(self, messages: list, on_reflect) -> tuple[str, str]:
        user_question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        tool_outputs = "\n".join(
            f"- {m.get('tool_name', 'tool')}: {str(m.get('content', ''))[:400]}"
            for m in messages if m.get("role") == "tool"
        )
        if not tool_outputs:
            return "sufficient", ""

        prompt = (
            f"Original request: '{user_question}'\n"
            f"Tool results gathered so far:\n{tool_outputs}\n\n"
            "Do these results sufficiently answer the request? "
            'Return ONLY JSON: {"verdict": "sufficient"|"insufficient", "gap": "<what is missing, or empty>"}'
        )
        try:
            res = self._llm.chat_once(
                messages=[{"role": "user", "content": prompt}],
                options=self._settings.planning_options,
            )
            raw = re.sub(r"^```(?:json)?|```$", "", res.cleaned_content.strip(), flags=re.MULTILINE).strip()
            parsed = json.loads(raw)
            verdict = "insufficient" if str(parsed.get("verdict", "")).lower() == "insufficient" else "sufficient"
            gap = str(parsed.get("gap", ""))
        except Exception as exc:
            print(f"[Thinking] Critique step failed ({exc}); assuming results are sufficient.")
            verdict, gap = "sufficient", ""

        if on_reflect:
            on_reflect(verdict, gap)
        return verdict, gap


# Backward-compat alias: earlier flat-loop agent, superseded by
# PlanExecuteReflectAgent above.
Agent = PlanExecuteReflectAgent
ToolRegistry = ScalableToolRegistry


# ==============================================================================
# CAMERA & CURIOSITY MANAGERS
# ==============================================================================

# Cosmetic-only: corner-bracket "focus frame" drawn over the live preview
# when a vision query comes in, so the user can see at a glance what part
# of the frame the assistant is looking at. Purely a UI overlay on the
# cv2.imshow preview — the frame bytes actually sent to the model
# (capture_jpeg_bytes) are never touched, so nothing changes downstream.
_FOCUS_FRAME_COLOR = (219, 91, 59)  # BGR, matches the blue reticle style
_FOCUS_FRAME_THICKNESS = 3

# Normalized (x0, y0, x1, y1) boxes, fraction of frame width/height.
# Used as: (a) the drawn box when no object detector is available or no
# detection matches, and (b) a spatial filter to disambiguate which
# detection the user means ("the person on the right").
_REGION_BOXES: dict[str, tuple[float, float, float, float]] = {
    "full":         (0.03, 0.03, 0.97, 0.97),
    "center":       (0.30, 0.25, 0.70, 0.75),
    "left":         (0.03, 0.15, 0.42, 0.85),
    "right":        (0.58, 0.15, 0.97, 0.85),
    "top":          (0.15, 0.03, 0.85, 0.42),
    "bottom":       (0.15, 0.58, 0.85, 0.97),
    "top-left":     (0.03, 0.03, 0.45, 0.45),
    "top-right":    (0.55, 0.03, 0.97, 0.45),
    "bottom-left":  (0.03, 0.55, 0.45, 0.97),
    "bottom-right": (0.55, 0.55, 0.97, 0.97),
}

# Ordered so compound directions ("top left") are matched before their
# single-word components ("top", "left").
_COMPOUND_REGION_PATTERNS = (
    (re.compile(r"\b(top|upper)[\s-]left\b", re.IGNORECASE), "top-left"),
    (re.compile(r"\b(top|upper)[\s-]right\b", re.IGNORECASE), "top-right"),
    (re.compile(r"\b(bottom|lower)[\s-]left\b", re.IGNORECASE), "bottom-left"),
    (re.compile(r"\b(bottom|lower)[\s-]right\b", re.IGNORECASE), "bottom-right"),
)
_SIMPLE_REGION_PATTERNS = (
    (re.compile(r"\bleft\b", re.IGNORECASE), "left"),
    (re.compile(r"\bright\b", re.IGNORECASE), "right"),
    (re.compile(r"\b(center|centre|middle)\b", re.IGNORECASE), "center"),
    (re.compile(r"\b(top|upper)\b", re.IGNORECASE), "top"),
    (re.compile(r"\b(bottom|lower)\b", re.IGNORECASE), "bottom"),
)


def infer_focus_region(user_question: str) -> str:
    """Cheap keyword heuristic: does the question point at a specific area
    of the frame, or ask about the image as a whole? Used both as the
    fallback box when detection is unavailable, and to disambiguate which
    detected object the user means when several match.
    """
    for pattern, region in _COMPOUND_REGION_PATTERNS:
        if pattern.search(user_question):
            return region
    for pattern, region in _SIMPLE_REGION_PATTERNS:
        if pattern.search(user_question):
            return region
    return "full"


# ------------------------------------------------------------------
# Object detection: sizes the focus-frame to the actual object instead
# of a fixed fractional guess. Additive only — if the model files are
# missing, ObjectDetector.available is False and callers fall back to
# the region-box behavior above; nothing else in the pipeline depends
# on this being present.
# ------------------------------------------------------------------

_VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Everyday words the user might say, mapped to the detector's class names.
_OBJECT_KEYWORD_ALIASES = {
    "human": "person", "man": "person", "woman": "person", "people": "person", "guy": "person",
    "vehicle": "car", "truck": "car", "automobile": "car",
    "tv": "tvmonitor", "screen": "tvmonitor", "monitor": "tvmonitor", "television": "tvmonitor",
    "plant": "pottedplant", "flower": "pottedplant",
    "bike": "bicycle", "motorcycle": "motorbike",
    "mug": "bottle", "cup": "bottle", "glass": "bottle", "jar": "bottle",
    "laptop": "tvmonitor", "computer": "tvmonitor", "phone": "tvmonitor",
    "cellphone": "tvmonitor", "mobile": "tvmonitor", "tablet": "tvmonitor",
    "animal": "dog", "pet": "dog", "cat": "cat", "puppy": "dog", "kitten": "cat",
    "table": "diningtable", "desk": "diningtable",
    "sofa": "sofa", "couch": "sofa",
    "chair": "chair", "stool": "chair",
    "bottle": "bottle", "can": "bottle",
    "bird": "bird", "fish": "bird",
    "bus": "bus", "van": "car",
    "train": "train",
}

# Only these get their real detected label surfaced anywhere in logs;
# every other VOC class is treated as a generic "object" match.
_NAMED_CLASSES = {"person", "car"}


def infer_focus_object_keyword(user_question: str) -> str | None:
    """Does the question name a specific kind of object ('the car',
    'that dog')? Returns the detector's class name for it, or None if the
    question is purely spatial ('the thing on the right') or generic
    ('describe the image')."""
    lowered = user_question.lower()
    for alias, canonical in _OBJECT_KEYWORD_ALIASES.items():
        if re.search(rf"\b{alias}\b", lowered):
            return canonical
    for cls in _VOC_CLASSES[1:]:
        if re.search(rf"\b{cls}\b", lowered):
            return cls
    return None


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixel coords


class BaseObjectDetector(ABC):
    """Interface for anything that can localize objects in a frame. The
    rest of the pipeline (CameraManager, resolve_focus_box, the live
    overlay loop) only depends on this contract, not on any particular
    backend — swapping MobileNet-SSD for YOLO or Grounding DINO later is
    a matter of writing a new subclass, not touching call sites.
    """

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether the backend loaded successfully and can be used."""

    @abstractmethod
    def detect(self, frame) -> list["Detection"]:
        """Returns spatial localizations only (label + confidence + box).
        Must never be used for natural-language description — that stays
        the VLM's job; this only answers 'where'."""


class ObjectDetector(BaseObjectDetector):
    """Thin wrapper around a lightweight OpenCV DNN (MobileNet-SSD) used
    only to size the focus-frame to the object the user asked about.
    Loading is best-effort: missing model files disable detection rather
    than crashing the app, since this is a UX enhancement, not a
    dependency of the core assistant loop."""

    def __init__(self, prototxt_path: str, model_path: str, min_confidence: float = 0.45):
        self._min_confidence = min_confidence
        self._net = None
        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            try:
                self._net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            except Exception as exc:
                print(f"[ObjectDetector] Failed to load model, focus-frame will fall back to region boxes: {exc}")
        else:
            print(
                f"[ObjectDetector] Model files not found ({prototxt_path!r} / {model_path!r}); "
                "focus-frame will fall back to region boxes. See README for download instructions."
            )

    @property
    def available(self) -> bool:
        return self._net is not None

    def detect(self, frame) -> list[Detection]:
        if self._net is None:
            return []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self._net.setInput(blob)
        raw = self._net.forward()

        detections: list[Detection] = []
        for i in range(raw.shape[2]):
            confidence = float(raw[0, 0, i, 2])
            if confidence < self._min_confidence:
                continue
            class_id = int(raw[0, 0, i, 1])
            if class_id <= 0 or class_id >= len(_VOC_CLASSES):
                continue
            box = raw[0, 0, i, 3:7] * [w, h, w, h]
            x0, y0, x1, y1 = box.astype(int)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            detections.append(Detection(label=_VOC_CLASSES[class_id], confidence=confidence, box=(x0, y0, x1, y1)))
        return detections


def resolve_focus_box(
    detections: list[Detection], region: str, object_keyword: str | None, frame_w: int, frame_h: int
) -> tuple[int, int, int, int] | None:
    """Picks the single detection that best matches what the user asked
    about: filtered first by object type if one was named, then by
    on-screen location if given, then by confidence. Returns None if
    nothing detected matches, so the caller can fall back to the region
    box instead of drawing a wrong/empty frame."""
    candidates = detections
    if object_keyword:
        candidates = [d for d in candidates if d.label == object_keyword]
        if not candidates:
            # Asked for a specific object type that wasn't detected at all —
            # don't box something unrelated; let the caller fall back to
            # the plain region box instead.
            return None

    if region != "full" and region in _REGION_BOXES:
        rx0, ry0, rx1, ry1 = _REGION_BOXES[region]

        def _center_in_region(d: Detection) -> bool:
            cx = ((d.box[0] + d.box[2]) / 2) / frame_w
            cy = ((d.box[1] + d.box[3]) / 2) / frame_h
            return rx0 <= cx <= rx1 and ry0 <= cy <= ry1

        region_matched = [d for d in candidates if _center_in_region(d)]
        if region_matched:
            candidates = region_matched

    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence).box


def draw_focus_frame(frame, box: tuple[int, int, int, int], color=_FOCUS_FRAME_COLOR, thickness: int = _FOCUS_FRAME_THICKNESS) -> None:
    """Draws four rounded L-shaped corner brackets around `box`, in place,
    matching the reticle/focus-frame look used elsewhere in the UI."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    corner_len = max(10, int(min(w, h) * 0.18))
    r = min(14, corner_len // 2)

    def _corner(cx: int, cy: int, dx: int, dy: int) -> None:
        cv2.line(frame, (cx + dx * r, cy), (cx + dx * corner_len, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + dy * r), (cx, cy + dy * corner_len), color, thickness, cv2.LINE_AA)
        start_angle = {(1, 1): 180, (-1, 1): 270, (1, -1): 90, (-1, -1): 0}[(dx, dy)]
        cv2.ellipse(frame, (cx + dx * r, cy + dy * r), (r, r), 0, start_angle, start_angle + 90, color, thickness, cv2.LINE_AA)

    _corner(x0, y0, 1, 1)     # top-left
    _corner(x1, y0, -1, 1)    # top-right
    _corner(x0, y1, 1, -1)    # bottom-left
    _corner(x1, y1, -1, -1)   # bottom-right


@dataclass
class PerceptionSnapshot:
    """Continuously-maintained scene state, refreshed by the standing
    perception loop rather than recomputed inside a user turn. `gist` is
    a cheap, comparable fingerprint of what's currently in view (sorted
    label:count pairs) — good enough to detect "something changed"
    without needing a VLM call just to watch the room."""

    timestamp: float
    detections: list["Detection"]
    gist: tuple[str, ...]

    def as_prompt_line(self) -> str:
        if not self.gist:
            return "Nothing currently recognized in view."
        return "Currently in view: " + ", ".join(self.gist) + "."


@dataclass
class PerceptionEvent:
    """A stable scene change, handed off from the perception thread to the
    asyncio event loop. Consumed by WorkflowController.handle_scene_event_async,
    which decides — gated by a cooldown, not on every change — whether it's
    worth a proactive, unprompted comment."""

    snapshot: PerceptionSnapshot
    previous_gist: tuple[str, ...]


def _detections_to_gist(detections: list["Detection"]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    return tuple(sorted(f"{label} x{n}" if n > 1 else label for label, n in counts.items()))


# ==============================================================================
# WEB INTERACTION TOOL (Gemini Computer Use + Playwright)
# ==============================================================================

def browse_website(task: str, start_url: str | None = None) -> str:
    """Interact with a website to accomplish a task using Gemini Computer Use
    and Playwright.

    The model sees screenshots of the browser and returns actions (click, type,
    scroll, navigate, etc.) which are executed client-side via Playwright. The
    loop continues until the model decides the task is complete or the step
    limit is reached.

    Args:
        task: Natural language description of what to accomplish on the web.
        start_url: Optional URL to open first. Defaults to Google if omitted.
    """
    from playwright.sync_api import sync_playwright

    # Lazy-import genai to avoid circular-import issues at module load time;
    # the module-level `from google import genai` already populated the
    # namespace, but a local import here makes the dependency explicit and
    # lets the function work even if the top-level import order changes.
    client = genai.Client()
    settings = _get_runtime_settings()

    model_name = settings.browser_interaction_model
    vw = settings.browser_viewport_width
    vh = settings.browser_viewport_height
    max_steps = settings.browser_max_steps
    headless = settings.browser_headless
    default_url = start_url or settings.browser_default_url

    steps_log: list[str] = []
    final_summary = ""

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": vw, "height": vh})

        # Navigate to the starting URL
        try:
            page.goto(default_url, wait_until="domcontentloaded", timeout=15000)
        except Exception as exc:
            browser.close()
            return f"TOOL_ERROR: Failed to open {default_url}: {exc}"

        # Bootstrap the Gemini Computer Use interaction
        try:
            interaction = client.interactions.create(
                model=model_name,
                input=f"Task: {task}\n\nComplete this task step by step using the browser.",
                tools=[{
                    "type": "computer_use",
                    "environment": "browser",
                    "enable_prompt_injection_detection": True,
                }],
            )
        except Exception as exc:
            browser.close()
            return f"TOOL_ERROR: Gemini Computer Use init failed: {exc}"

        for step_num in range(1, max_steps + 1):
            # Inspect the latest interaction for model_output blocks
            model_blocks = []
            for interaction_step in interaction.steps:
                if interaction_step.type == "model_output" and hasattr(interaction_step, "content"):
                    for block in interaction_step.content:
                        if block.type == "function_call":
                            model_blocks.append(block)
                        elif block.type == "text" and hasattr(block, "text"):
                            # Model emitted a text summary — capture it
                            final_summary = block.text

            if not model_blocks:
                # Model produced no more actions — task is done
                break

            # Execute every action the model requested in this round
            for block in model_blocks:
                name = block.function_call.name
                args = block.function_call.args
                action_desc = f"{name}({args})"
                steps_log.append(f"Step {step_num}: {action_desc}")

                try:
                    if name == "navigate_to_url":
                        page.goto(args["url"], wait_until="domcontentloaded", timeout=15000)
                    elif name == "click":
                        x = int(args["x"] / 1000.0 * vw)
                        y = int(args["y"] / 1000.0 * vh)
                        page.mouse.click(x, y)
                        page.wait_for_load_state("domcontentloaded", timeout=10000)
                    elif name == "type":
                        page.keyboard.type(args["text"], delay=20)
                    elif name == "scroll":
                        magnitude = args.get("magnitude_in_pixels", 300)
                        page.mouse.wheel(0, magnitude)
                    elif name == "key":
                        page.keyboard.press(args["key"])
                    elif name == "go_back":
                        page.go_back(wait_until="domcontentloaded", timeout=10000)
                    elif name == "go_forward":
                        page.go_forward(wait_until="domcontentloaded", timeout=10000)
                    elif name == "screenshot":
                        pass  # No-op — screenshot is taken below anyway
                    else:
                        steps_log.append(f"  [skipped unknown action: {name}]")
                        continue
                except Exception as exc:
                    steps_log.append(f"  [execution error: {exc}]")

                # Small delay so the page can settle after the action
                page.wait_for_timeout(500)

            # Take a screenshot and feed it back to Gemini
            try:
                screenshot_bytes = page.screenshot(type="png")
            except Exception as exc:
                steps_log.append(f"  [screenshot error: {exc}]")
                break

            try:
                interaction = client.interactions.create(
                    model=model_name,
                    previous_interaction_id=interaction.id,
                    input=[genai.types.Part.from_bytes(screenshot_bytes, mime_type="image/png")],
                    tools=[{
                        "type": "computer_use",
                        "environment": "browser",
                        "enable_prompt_injection_detection": True,
                    }],
                )
            except Exception as exc:
                steps_log.append(f"  [Gemini follow-up error: {exc}]")
                break

        # Capture the final page state for the summary
        try:
            final_url = page.url
            final_title = page.title()
        except Exception:
            final_url = "(unknown)"
            final_title = "(unknown)"

        browser.close()

    # Build a concise result string for the agent
    action_count = len(steps_log)
    if final_summary:
        result = final_summary
    elif action_count > 0:
        result = (
            f"Browser task completed after {action_count} actions. "
            f"Final page: '{final_title}' at {final_url}. "
            f"Actions taken: {'; '.join(steps_log[-5:])}"
        )
    else:
        result = "Browser task completed with no recorded actions."

    return result


# Module-level holder so _browse_website can access Settings without being a
# class.  Set once at startup by build_assistant(); avoids threading a Settings
# reference through the ollama tool-call interface (which only passes the
# declared function signature).
_runtime_settings: Settings | None = None


def _get_runtime_settings() -> Settings:
    if _runtime_settings is None:
        raise RuntimeError("WebInteractionTool: runtime Settings not initialized")
    return _runtime_settings


def set_runtime_settings(settings: Settings) -> None:
    global _runtime_settings
    _runtime_settings = settings


class CameraManager:
    """Two independent background loops, deliberately kept separate:

    - `_loop` (unchanged from v1.8): raw frame grab + live preview window.
      Always runs once `start()` is called; this is what keeps
      `capture_jpeg_bytes()` fresh for a query, and was already
      continuous — it was never the bottleneck.
    - `_perception_loop` (new): runs object detection on a timer,
      independent of whether a user turn is happening, and maintains
      `latest_perception`. A detection result has to hold stable across
      `perception_stability_polls` consecutive polls before it's treated
      as a real scene change (debounces single-frame noise) and handed to
      the asyncio event loop as a `PerceptionEvent`.

    This is what turns vision from "fires per-turn on explicit query"
    into a standing scene monitor: detection is always warm, a query
    never pays fresh detection latency, and a sustained change in the
    room can initiate a turn instead of only ever responding to one.
    """

    def __init__(self, device_index: int = 0, object_detector: "BaseObjectDetector | None" = None):
        self._camera = cv2.VideoCapture(device_index)
        self._last_frame = None
        self._running = True
        self._thread: threading.Thread | None = None
        self._perception_thread: threading.Thread | None = None
        self._detector = object_detector
        self._focus_box: tuple[int, int, int, int] | None = None
        self._focus_expiry: float = 0.0
        self.latest_perception: PerceptionSnapshot | None = None

    @property
    def last_frame(self):
        return self._last_frame

    @property
    def is_streaming(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._camera.isOpened():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def start_perception_monitor(
        self,
        loop: asyncio.AbstractEventLoop,
        event_queue: "asyncio.Queue",
        settings: Settings,
    ) -> None:
        """Starts the standing scene-monitor thread. Runs independent of
        `start()` / the video-preview thread, and independent of whether
        any user turn is in progress. Safe no-op if continuous perception
        is disabled in settings or no detector is available."""
        if not settings.enable_continuous_perception or self._detector is None or not self._detector.available:
            return
        self._perception_thread = threading.Thread(
            target=self._perception_loop, args=(loop, event_queue, settings), daemon=True
        )
        self._perception_thread.start()

    def _perception_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        event_queue: "asyncio.Queue",
        settings: Settings,
    ) -> None:
        stable_gist: tuple[str, ...] | None = None
        pending_gist: tuple[str, ...] | None = None
        pending_streak = 0

        while self._running:
            time.sleep(settings.perception_poll_interval_seconds)
            frame = self._last_frame
            if frame is None:
                continue

            detections = self._detector.detect(frame)
            gist = _detections_to_gist(detections)
            snapshot = PerceptionSnapshot(timestamp=time.time(), detections=detections, gist=gist)
            self.latest_perception = snapshot  # always kept warm, regardless of stability/events

            if gist == pending_gist:
                pending_streak += 1
            else:
                pending_gist, pending_streak = gist, 1

            if pending_streak >= settings.perception_stability_polls and gist != stable_gist:
                event = PerceptionEvent(snapshot=snapshot, previous_gist=stable_gist or ())
                stable_gist = gist
                try:
                    loop.call_soon_threadsafe(event_queue.put_nowait, event)
                except RuntimeError:
                    # Event loop already closed (shutdown race) — drop silently.
                    pass

    def set_focus_target(self, region: str, object_keyword: str | None = None, duration: float = 6.0) -> None:
        """Sizes and positions the focus-frame for the current turn.
        Tries object detection first so the frame fits the actual object
        ('the car', 'the person on the right'); falls back to the plain
        fractional region box if no detector is available or nothing
        detected matches. Cosmetic only — does not affect what gets
        captured/sent to the model."""
        frame = self._last_frame
        box = None
        detections = []
        if frame is not None and self._detector is not None and self._detector.available:
            h, w = frame.shape[:2]
            detections = self._detector.detect(frame)
            box = resolve_focus_box(detections, region, object_keyword, w, h)

        if box is None and detections:
            best = max(detections, key=lambda d: d.confidence)
            box = best.box

        if box is None and frame is not None:
            h, w = frame.shape[:2]
            fx0, fy0, fx1, fy1 = _REGION_BOXES.get(region, _REGION_BOXES["full"])
            box = (int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h))

        self._focus_box = box
        self._focus_expiry = time.time() + duration

    def _loop(self) -> None:
        while self._running:
            success, frame = self._camera.read()
            if success:
                self._last_frame = frame  # unannotated frame; this is what capture_jpeg_bytes sends to the model

                display_frame = frame
                if self._focus_box:
                    if time.time() < self._focus_expiry:
                        display_frame = frame.copy()
                        draw_focus_frame(display_frame, self._focus_box)
                    else:
                        self._focus_box = None

                cv2.imshow("P.A.L. Vision Feed", display_frame)
                if cv2.waitKey(1) == 27:
                    self._running = False
                    break

    def capture_jpeg_bytes(self) -> bytes | None:
        if self._last_frame is None:
            return None
        success, buf = cv2.imencode(".jpg", self._last_frame)
        return buf.tobytes() if success else None

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join()
        if self._perception_thread:
            self._perception_thread.join()
        self._camera.release()
        cv2.destroyAllWindows()



class CuriosityPolicy:
    """Two independent triggers, deliberately kept separate:

    - `should_ask`: the existing fact-gathering curiosity (learn the
      user's preferences/workflows when little is known yet).
    - `detect_emotional_disclosure`: a distinct trigger for emotionally
      significant statements (loss, breakup, stress, grief...), where the
      right move isn't "ask a personalizing question" but "react like a
      friend would and leave space to talk about it if they want to."
    Kept as a cheap regex heuristic rather than another LLM round-trip,
    consistent with how `should_ask` is implemented; expand the pattern
    list as real conversations surface gaps.
    """

    _EMOTIONAL_SIGNAL_PATTERN = re.compile(
        r"\b("
        r"lost my|broke up|break up|breakup|divorc\w*|passed away|died|death of|"
        r"lonely|depress\w*|anxious|anxiety|stressed|stressful|hard time|"
        r"heartbroken|miscarriage|fired from|laid off|grieving|grief|"
        r"can't sleep|not okay|not ok|struggling|overwhelmed"
        r")\b",
        re.IGNORECASE,
    )

    def __init__(self, settings: Settings):
        self._settings = settings

    def should_ask(self, query: str, total_entries: int) -> bool:
        return total_entries < 3 and len(query.split()) > 3

    def detect_emotional_disclosure(self, query: str) -> bool:
        return bool(self._EMOTIONAL_SIGNAL_PATTERN.search(query))


# ==============================================================================
# CONTEXT MANAGER & WORKFLOW CONTROLLER
# ==============================================================================

ASSISTANT_NAME = "P.A.L."

PERSONALITY_PROMPT = (
    f"Your name is {ASSISTANT_NAME}.\n"
    "You are an advanced personal AI assistant inspired by the qualities of a high-performance executive aide.\n"
    "Your primary purpose is to make the user's life easier by providing accurate information, anticipating needs, organizing complexity, and assisting with decisions.\n"
    "You are calm, intelligent, precise, and composed under all circumstances.\n"
    "Communicate with confidence and clarity. Avoid unnecessary verbosity, but provide depth when the problem requires it.\n"
    "You are not a generic chatbot or customer support agent. You behave like a trusted expert assistant who understands context and priorities.\n"
    "Maintain a professional but natural conversational style. You can use subtle dry humor occasionally when appropriate, but never at the expense of usefulness or respect.\n"
    "Challenge incorrect assumptions politely and directly. Your goal is not to agree, but to improve the quality of the user's decisions.\n"
    "When the user is exploring complex ideas, engage intellectually and help refine their thinking.\n"
    "When the user needs execution, be concise and action-oriented.\n"
    "Avoid unnecessary greetings, filler phrases, excessive enthusiasm, or artificial emotional expressions.\n"
    "Your personality should feel like a highly capable partner: observant, reliable, and discreet.\n"
    "Always respond in the same language the user just wrote in; default to English if the language is unclear or ambiguous.\n"
    "If the task or the question is non clear or ambiguos, you must make a single answer to the user to clarify the task of the question you will respond to.\n"
    "If nor the Internet source nor the internal memory source is able to give an updated answer you are authorized to return a simple 'I do not know how to answer either, Sorry'"
)

# Adapted from v1.4's MEMORY_USAGE_PROMPT_TEMPLATE to describe the hybrid
# retrieval shape (vector matches + typed/categorized graph connections)
# instead of a flat, undifferentiated fact list, and to keep the temporal/
# location-query directive that v1.5 previously hardcoded separately.
MEMORY_USAGE_PROMPT_TEMPLATE = (
    "You have access to relevant information about the user, retrieved in two stages:\n"
    "an episodic similarity match against stored memories, then a semantic graph expansion around the "
    "entities involved, grouped by category and typed (PERSON, OBJECT, LOCATION, PREFERENCE, etc.):\n"
    "{facts}\n"
    "Use this information only when it improves accuracy, relevance, or personalization.\n"
    "Each graph connection is marked CURRENT or EXPIRED with a timestamp; prioritize CURRENT facts, "
    "and prefer the most recent valid_from over older or expired ones when they conflict.\n"
    "When answering temporal or location queries (e.g. 'where are my keys'), rely on the graph "
    "connections to state the exact last timestamp and location observed, not a guess.\n"
    "Do not mention the existence of stored information, memory systems, retrieval processes, or previous conversations.\n"
    "Do not reveal private stored details unless the user explicitly asks for that information.\n"
    "Use context naturally, as an assistant familiar with the user's ongoing objectives and preferences."
)

VISION_BEHAVIOR_PROMPT = (
    "The user may provide visual information through a camera.\n"
    "Treat visual information as additional context available for reasoning.\n"
    "Do not describe images unnecessarily.\n"
    "Analyze or reference visual information only when it helps answer the user's request or when explicitly asked.\n"
    "Prioritize usefulness over observation."
)

# New: fed by the standing perception monitor (CameraManager._perception_loop),
# not by the per-turn image attachment. This is a coarse, low-confidence
# object-label gist maintained continuously in the background — distinct
# from the actual image frame attached to this turn (if any), which is
# authoritative and richer. Ambient gist is for passive continuity only.
AMBIENT_PERCEPTION_PROMPT_TEMPLATE = (
    "A background scene monitor maintains a rough, continuously-updated sense of what's physically nearby.\n"
    "{gist_line}\n"
    "This is a coarse label list, not a verified description — treat it as weak ambient context only.\n"
    "Never state it as fact, never lead with it unprompted, and never mention 'the scene monitor' or "
    "'background detection' to the user.\n"
    "Use it only if it's directly relevant to what the user just asked."
)

WEB_ACCESS_PROMPT = (
    "You have access to live internet tools.\n"
    "Use them when information may have changed or when current external data is required.\n"
    "Examples include news, market data, prices, availability, schedules, regulations, and recent developments.\n"
    "Do not use external tools for stable knowledge or when existing context is sufficient.\n"
    "For complex questions, you may need more than one tool call: search first, then fetch a specific page for detail, or run a second, differently-worded search if the first was too broad or too narrow. Reason about what you actually learned before deciding whether another call is needed or you already have enough to answer.\n"
    "If a tool result begins with TOOL_ERROR, the call failed or returned nothing useful. Do not repeat the identical call. Adjust your query, try a more specific or more general phrasing, or use a different tool. Only give up and answer from what you have after a genuinely different attempt has also failed.\n"
    "You also have access to a browser interaction tool that can navigate websites, click buttons, fill forms, and perform multi-step web tasks using a live browser session. Use this when the task requires interacting with a website (e.g. buying tickets, filling out a reservation form, subscribing to a service) rather than just reading information. Describe the full task you want accomplished, and optionally provide a starting URL.\n"
    "Attribution is mandatory, not optional: every piece of information that came from a tool call must be attributed by name in your spoken answer, e.g. 'according to 3B Meteo' or 'Reuters reports that'. Never present web-sourced information as if it were your own unprompted knowledge.\n"
    "If you checked multiple sources and they agree, say so explicitly and generally, e.g. 'I checked multiple sources and they all agree that...', rather than naming each one individually.\n"
    "If sources disagree, name the conflicting sources and state the discrepancy rather than silently picking one.\n"
    "Weave attribution into a normal spoken sentence, never as a bracketed citation, link, or footnote."
)

SPEECH_OUTPUT_PROMPT = (
    "Your entire response will be converted to speech by a text-to-speech engine and read aloud; the user will never see the raw text.\n"
    "Write only in plain, natural spoken sentences, exactly as a person would say them out loud.\n"
    "Never use markdown: no asterisks, no bold, no italics, no headers, no bullet points, no numbered lists, no tables.\n"
    "Never use symbols that are not naturally spoken: no dashes used as bullets, no colons introducing a list, no parentheses for asides, no quotation marks, no emojis.\n"
    "If you need to present multiple items, say them as a flowing sentence using words like 'first, second, and third' or 'one option is... another is...', not as a list.\n"
    "Spell out things the way they are spoken, not the way they are written: say 'ten percent' rather than '10%', 'the U.A.E.' rather than 'UAE' if it is meant to be read as separate letters, and dates as they'd be spoken aloud.\n"
    "Do not include section titles, labels, or headings of any kind. Just speak the answer as continuous natural prose."
)

CURIOSITY_PROMPT = (
    "Your objective is to gradually understand the user's preferences, goals, workflows, and priorities.\n"
    "Ask a personalizing question only when it creates clear future value.\n"
    "Never interrupt an active task with unnecessary questions.\n"
    "Limit curiosity questions to one short question when appropriate."
)

# Fixes the "summarizes the interaction back to the user" failure mode.
# The output is read aloud by a TTS engine, so a recap reads like a report,
# not a conversation. Push the model toward short, organic reactions.
NATURAL_ACKNOWLEDGEMENT_PROMPT = (
    "When the user shares new personal information, do not summarize, recap, or repeat back what they just told you.\n"
    "React the way a close friend would in a real conversation: short, organic acknowledgements like "
    "'Interesting.', 'Got it.', 'Ok, good to know.', 'Understood.', or 'Tell me more about that.'\n"
    "If you know the user's name from context, use it naturally and sparingly, e.g. 'Understood, Marco.', "
    "not in every sentence.\n"
    "Only go longer or more structured when the user is actually asking for detailed help, an explanation, "
    "or a decision; casual disclosures get a casual, brief reaction, not a report."
)

# New: the model is implicitly gathering information about the user over
# time; when that information is emotionally significant rather than a
# neutral fact (preference, project detail...), the right reaction is
# warmth and space to talk, not data collection.
EMOTIONAL_SUPPORT_PROMPT = (
    "The user just said something that sounds emotionally significant or difficult (a loss, stress, conflict, "
    "or similar).\n"
    "React first with genuine warmth, like a close friend would, not a clinician and not a customer service agent.\n"
    "If it feels natural, follow up with one gentle, open question such as 'Do you want to talk about it?' or "
    "'How are you holding up?' Do not interrogate or ask more than one follow-up question.\n"
    "Do not diagnose, label, or clinically analyze what they're feeling.\n"
    "Keep it brief and conversational, not a formal check-in script.\n"
    "If what they describe sounds like serious or ongoing distress, gently encourage them to talk to someone "
    "they trust or a professional, without being alarmist or repeating this every turn."
)

# New: image-attached turns where the user is asking to solve a written
# problem (math, an equation, an exercise) rather than just describe the
# scene. The base VISION_BEHAVIOR_PROMPT never told the model to actually
# transcribe and work through text it sees; this closes that gap and
# shapes the output as result-first, then the procedure, matching how a
# person would want a solved problem read back to them.
PROBLEM_SOLVING_PROMPT = (
    "The user is asking you to solve a problem shown in the image (a math problem, equation, exercise, or "
    "similar written or handwritten text).\n"
    "First read the problem carefully and exactly as written, including handwriting, before attempting it.\n"
    "State the final result clearly and early, in plain spoken language.\n"
    "Then walk through the solving procedure step by step, in the order you used it, as you would explain it "
    "out loud to someone learning it, not as a formal proof.\n"
    "If any part of the text or handwriting is unclear or ambiguous, say so plainly and state your best "
    "interpretation rather than guessing silently or skipping it."
)

CLARIFICATION_ANALYSIS_PROMPT = (
    "You are about to execute a task that requires tool use. Before proceeding, "
    "determine whether you have all the critical information needed to complete "
    "this task effectively.\n\n"
    "Consider the following based on the user's request and your available context:\n"
    "- Is the user's location clear and sufficient? (for location-dependent tasks)\n"
    "- Are budget constraints known? (for purchasing, booking, or comparison tasks)\n"
    "- Are time constraints known? (for scheduling or availability tasks)\n"
    "- Are there ambiguous aspects that would lead to a wrong result if assumed?\n"
    "- Are there personal preferences that would significantly change the approach?\n"
    "- Did the user already provide enough detail to proceed confidently?\n\n"
    "IMPORTANT: Only ask for information that is genuinely critical AND not already "
    "inferable from the available context (the user's own words, stored memory, "
    "location, time, or prior conversation). If the user already said 'quick and cheap' "
    "for example, do not ask about budget again.\n\n"
    "Return ONLY a JSON object, no prose, no markdown fences:\n"
    '{"proceed": true} if nothing critical is missing, OR\n'
    '{"proceed": false, "reasoning": "brief explanation of what is missing", '
    '"question": "natural, conversational clarifying question to ask the user"}'
)

_PROBLEM_SOLVING_PATTERN = re.compile(
    r"\b(solve|solution|work(?:\s+it|s)?\s+out|calculate|compute|equation|exercise|homework|"
    r"what'?s the answer|figure out)\b",
    re.IGNORECASE,
)


def is_problem_solving_query(user_question: str) -> bool:
    """Heuristic gate for PROBLEM_SOLVING_PROMPT: only meaningful when an
    image is attached (checked by the caller) and the question itself
    signals 'solve this', not just 'what is this'."""
    return bool(_PROBLEM_SOLVING_PATTERN.search(user_question))


class ContextManager:
    """Assembles the per-turn system prompt from modular sections. Only
    knows about section *content* and conditional inclusion rules; retrieval
    (memory) and scoring (curiosity) are computed by their own components
    and passed in as plain values."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def _build_situational_awareness_prompt(self) -> str:
        """Current time + location, recomputed every turn. This is what
        lets 'what time is it' or 'what's the weather here' be answered
        directly (or with a location-scoped search query) instead of the
        model guessing or refusing for lack of context.
        """
        if self._settings.user_timezone:
            try:
                now = datetime.now(ZoneInfo(self._settings.user_timezone))
            except Exception:
                now = datetime.now()
        else:
            now = datetime.now()
        formatted = now.strftime("%A, %d %B %Y, %H:%M")
        tz_label = f" ({self._settings.user_timezone})" if self._settings.user_timezone else ""

        coords = ""
        if self._settings.user_latitude is not None and self._settings.user_longitude is not None:
            coords = f" (lat {self._settings.user_latitude:.4f}, lon {self._settings.user_longitude:.4f})"

        return (
            f"Current date and time: {formatted}{tz_label}.\n"
            f"The user's current location is: {self._settings.user_location_name}{coords}.\n"
            "Use this directly to answer questions about the current time, date, day of the week, or the "
            "user's location, without needing a tool call and without treating it as uncertain.\n"
            "Resolve relative references ('today', 'tomorrow', 'this weekend', 'near me', 'from here') against "
            "this date and location before answering, or before formulating a web search query."
        )

    def build_system_prompt(
        self,
        memory_context: str,
        curiosity: bool,
        emotional_support: bool = False,
        problem_solving: bool = False,
        ambient_perception: "PerceptionSnapshot | None" = None,
    ) -> str:
        sections = [
            PERSONALITY_PROMPT,
            self._build_situational_awareness_prompt(),
            SPEECH_OUTPUT_PROMPT,
            NATURAL_ACKNOWLEDGEMENT_PROMPT,
        ]
        if memory_context:
            sections.append(MEMORY_USAGE_PROMPT_TEMPLATE.format(facts=memory_context))
        sections.append(VISION_BEHAVIOR_PROMPT)
        if ambient_perception is not None:
            sections.append(AMBIENT_PERCEPTION_PROMPT_TEMPLATE.format(gist_line=ambient_perception.as_prompt_line()))
        if problem_solving:
            sections.append(PROBLEM_SOLVING_PROMPT)
        sections.append(WEB_ACCESS_PROMPT)
        if emotional_support:
            sections.append(EMOTIONAL_SUPPORT_PROMPT)
        if curiosity:
            sections.append(CURIOSITY_PROMPT)
        return "\n\n".join(sections)


class WorkflowController:
    def __init__(
        self,
        llm_client: LLMClient,
        router: Router,
        agent: Agent,
        memory_manager: HybridMemoryManager,
        working_memory: WorkingMemoryBuffer,
        camera_manager: CameraManager,
        context_manager: ContextManager,
        curiosity_policy: CuriosityPolicy,
        settings: Settings,
    ):
        self._llm = llm_client
        self._router = router
        self._agent = agent
        self._memory = memory_manager
        self._working_memory = working_memory
        self._camera = camera_manager
        self._context = context_manager
        self._curiosity = curiosity_policy
        self.settings = settings

    # ------------------------------------------------------------------
    # Async entrypoints — the runtime (main()) is now an asyncio event
    # loop with two independent event producers (user text, standing
    # perception monitor) feeding one queue. Neither LLMClient nor
    # ArcadeDBClient/Chroma were rewritten onto native async I/O — that
    # would mean replacing the ollama/requests/chromadb SDKs, a much
    # larger and riskier change. Instead, each blocking call site is
    # pushed onto a worker thread via asyncio.to_thread, which is enough
    # to make the *application loop* concurrent and non-blocking: a scene
    # event can be evaluated, and the next user keystroke can be read,
    # while a previous turn's LLM call is still in flight. This is the
    # standard incremental path from a synchronous codebase to an
    # event-driven one; it is not a claim that every I/O call below is
    # natively async.
    # ------------------------------------------------------------------
    async def handle_turn_async(self, user_question: str) -> None:
        await asyncio.to_thread(self._handle_turn_sync, user_question)

    # ------------------------------------------------------------------
    # Core turn handling (blocking; always run via asyncio.to_thread from
    # handle_turn_async above, never called directly from the event loop)
    # ------------------------------------------------------------------
    def _handle_turn_sync(self, user_question: str) -> None:
        # 1. Retrieve hybrid memory context (Vector Filter + Graphiti Temporal Expansion)
        memory_context = self._memory.query(user_question)

        # 2. Snapshot the camera (if available) before building the prompt,
        # so vision-dependent prompt sections (problem-solving) can be
        # decided with knowledge of whether an image is actually attached.
        # This is still an on-demand capture of the *authoritative* frame
        # for this turn (sent to the model) — separate from, and richer
        # than, the ambient gist below, which comes from the standing
        # perception monitor and was already warm before this turn began.
        img_bytes = self._camera.capture_jpeg_bytes()
        include_problem_solving = bool(img_bytes) and is_problem_solving_query(user_question)
        ambient_perception = self._camera.latest_perception

        # 3. Build Prompt & Construct Messages
        include_curiosity = self._curiosity.should_ask(user_question, self._memory.total_entry_count())
        include_emotional_support = self._curiosity.detect_emotional_disclosure(user_question)
        system_prompt = self._context.build_system_prompt(
            memory_context, include_curiosity, include_emotional_support, include_problem_solving,
            ambient_perception,
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._working_memory.as_messages())
        messages.append({"role": "user", "content": user_question})

        if img_bytes:
            messages[-1]["images"] = [img_bytes]
            # Cosmetic only: highlight what part of the frame / which object
            # the question refers to on the live preview. Tries to fit the
            # frame to the actual detected object; falls back to a region
            # box if detection is unavailable or nothing matches. Does not
            # affect the image bytes above or anything the model receives.
            self._camera.set_focus_target(
                infer_focus_region(user_question), infer_focus_object_keyword(user_question)
            )

        # 4. Classify Route (+ Complexity Tier + Tool Domains for PLAN) & Execute
        decision = self._router.classify_advanced(user_question)
        domain_note = f" domains={[d.value for d in decision.target_domains]}" if decision.target_domains else ""
        complexity_note = f" complexity={decision.complexity.value}" if decision.route == Route.PLAN else ""
        print(f"\n[Thinking] Route classified as {decision.route.value.upper()} ({decision.reason}){complexity_note}{domain_note}")
        if include_emotional_support:
            print("[Thinking] Emotional disclosure detected — responding with support framing this turn.")
        if include_problem_solving:
            print("[Thinking] Problem-solving request detected on an image — result-first, step-by-step framing this turn.")

        if decision.route == Route.PLAN:
            round_budget = (
                self.settings.single_step_max_rounds
                if decision.complexity == RouteComplexity.SINGLE_STEP
                else self.settings.max_agent_tool_rounds
            )

            def _on_thinking(round_num: int) -> None:
                if round_num == -1:
                    print("[Thinking] Analyzing information needs before acting...")
                else:
                    print(f"[Thinking] Reasoning step {round_num}/{round_budget}: deciding next action...")

            def _on_tool_call(name: str, arguments: dict) -> None:
                if name in ("web_search", "web_fetch"):
                    print(f"[Web searching] {name}({arguments})")
                elif name == "browse_website":
                    print(f"[Browser] {name}({arguments})")
                else:
                    print(f"[Tool Call] {name}({arguments})")

            def _on_plan(sub_goals: list) -> None:
                if sub_goals:
                    print(f"[Planning] Sub-goals: {sub_goals}")

            def _on_reflect(verdict: str, gap: str) -> None:
                if verdict == "insufficient":
                    print(f"[Reflecting] Gathered data insufficient — {gap}")
                else:
                    print("[Reflecting] Gathered data sufficient to answer.")

            def _on_clarification(question: str) -> None:
                print(f"[Thinking] Need to clarify with user before proceeding.")

            messages = self._agent.run(
                messages,
                domains=decision.target_domains,
                complexity=decision.complexity,
                on_tool_call=_on_tool_call,
                on_thinking=_on_thinking,
                on_plan=_on_plan,
                on_reflect=_on_reflect,
                on_clarification=_on_clarification,
            )

        # 5. Final LLM Answer Streaming
        stream = self._llm.chat_stream(messages, self.settings.final_answer_options)
        print(f"\n{self.settings.assistant_name}: ", end="")
        full_response = ""
        for chunk in stream:
            piece = chunk["message"]["content"]
            full_response += piece
            print(piece, end="", flush=True)
        print("\n")

        # 6. Record this turn in the in-process working-memory buffer
        # (Layer 1 -- never persisted) so the *next* turn has verbatim
        # continuity, independent of whatever episodic/semantic
        # retrieval surfaces in step 1. Camera context, if any, is passed
        # in explicitly so it stays scoped to this RAM-only buffer
        # instead of silently riding along into a Graph/Vector write.
        self._working_memory.add_turn(
            user_question, full_response,
            camera_context="Frame captured" if img_bytes else None,
        )

        # 7. Async Memory Extraction (Vector Store + ArcadeDB Temporal Graphiti Update).
        # extract_and_store returns whether anything was actually
        # persisted (GraphDB and/or VectorDB); if not, the turn stayed
        # RAM-only (already recorded in working memory above), so we log
        # that explicitly instead of leaving silence where a store
        # decision happened.
        def _extract_store_and_log(text: str, camera_ctx: str | None) -> None:
            persisted = self._memory.extract_and_store(text, camera_ctx)
            if not persisted:
                print(f"[RAM updated] (not persisted to VectorDB or GraphDB) {text[:80]!r}")

        threading.Thread(
            target=_extract_store_and_log,
            args=(user_question, "Frame captured" if img_bytes else None),
            daemon=True,
        ).start()


# ==============================================================================
# COMPOSITION ROOT & ENTRYPOINT
# ==============================================================================

def _time_conditional_greeting() -> str:
    hour = datetime.now().hour
    if 0 <= hour < 5:
        return f"{ASSISTANT_NAME}: Still up. Working on something, or just avoiding sleep?"
    if 5 <= hour < 7:
        return f"{ASSISTANT_NAME}: You're up early. This better be worth it."
    if 7 <= hour < 12:
        return f"{ASSISTANT_NAME}: Good morning."
    if 12 <= hour < 18:
        return f"{ASSISTANT_NAME}: Good afternoon."
    if 18 <= hour < 23:
        return f"{ASSISTANT_NAME}: Good evening."
    return f"{ASSISTANT_NAME}: Good night. Or morning, depending on how you look at it."


def build_assistant() -> tuple[WorkflowController, CameraManager]:
    settings = Settings.from_env()
    print(
        f"[Initializer] Situational context resolved: location={settings.user_location_name!r}, "
        f"timezone={settings.user_timezone!r}"
    )

    llm_client = LLMClient(settings)
    embedding_client = EmbeddingClient(settings)

    # Database Initialization
    chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    category_registry = CategoryRegistry(chroma_client, settings)
    arcadedb_client = ArcadeDBClient(settings)

    # Hybrid RAG Memory Integration
    hybrid_memory = HybridMemoryManager(
        llm_client, embedding_client, category_registry, arcadedb_client, settings
    )
    working_memory = WorkingMemoryBuffer(settings.working_memory_max_turns)

    tool_registry = ScalableToolRegistry()
    tool_registry.register(ToolDomain.WEB, "web_search", web_search, llm_client.raw_client.web_search)
    tool_registry.register(ToolDomain.WEB, "web_fetch", web_fetch, llm_client.raw_client.web_fetch)

    def query_memory(query: str) -> str:
        """Look up stored facts about the user, their preferences, contacts,
        or past conversations from long-term memory.

        Args:
            query: Natural language description of what to look up.
        """

    tool_registry.register(ToolDomain.MEMORY, "query_memory", query_memory, hybrid_memory.query)

    # Web Interaction tool (Gemini Computer Use + Playwright)
    set_runtime_settings(settings)
    tool_registry.register(ToolDomain.BROWSER, "browse_website", browse_website, browse_website)

    tool_executor = ToolExecutor(tool_registry)

    router = AdvancedRouter(llm_client, settings)
    agent = PlanExecuteReflectAgent(llm_client, tool_registry, tool_executor, settings)
    object_detector = ObjectDetector(
        settings.object_detector_prototxt_path,
        settings.object_detector_model_path,
        settings.object_detector_min_confidence,
    )
    camera_manager = CameraManager(object_detector=object_detector)
    context_manager = ContextManager(settings)
    curiosity_policy = CuriosityPolicy(settings)

    workflow = WorkflowController(
        llm_client, router, agent, hybrid_memory, working_memory,
        camera_manager, context_manager, curiosity_policy, settings
    )
    return workflow, camera_manager


async def _read_stdin_events(event_queue: "asyncio.Queue", prompt: str) -> None:
    """Text-input producer. Kept deliberately behind the same queue
    interface a future voice/AR input source would use (push a string
    onto event_queue) — swapping this for STT wake-word input later is a
    matter of adding another producer coroutine, not restructuring the
    dispatcher below.

    Known limitation of the CLI shape specifically: `input()` runs on a
    worker thread via asyncio.to_thread and blocks until Enter is pressed;
    that thread cannot be forcibly cancelled mid-read, so on shutdown it
    lingers until the next keystroke or process exit. Irrelevant once this
    producer is replaced by a real streaming input source (mic/AR input),
    which is the point of keeping it behind this interface.
    """
    while True:
        try:
            line = await asyncio.to_thread(input, prompt)
        except EOFError:
            await event_queue.put(None)  # sentinel: stop the dispatcher
            return
        text = line.strip()
        if text:
            await event_queue.put(text)


async def async_main() -> None:
    # 1. Pre-flight verification of environment variables
    ensure_environment_variables()

    # 2. Automatically check and initialize ArcadeDB container / engine readiness
    arcadedb_url = os.environ.get("ARCADEDB_URL", "http://localhost:2480")
    arcadedb_user = os.environ.get("ARCADEDB_USER", "root")
    arcadedb_pass = os.environ.get("ARCADEDB_PASSWORD", "playwithdata")
    ensure_arcadedb_running(arcadedb_url, user=arcadedb_user, password=arcadedb_pass)

    # 3. Build assistant components
    workflow, camera_manager = build_assistant()
    camera_manager.start()

    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    # Standing perception monitor (v1.9): runs independent of any turn and
    # feeds stable scene changes into the SAME queue as user text below.
    # This is the actual fix for "camera bolted on" — detection is no
    # longer something that only happens inside handle_turn.
    camera_manager.start_perception_monitor(loop, event_queue, workflow.settings)

    print(f"\n{_time_conditional_greeting()}\n")
    stdin_task = asyncio.create_task(
        _read_stdin_events(event_queue, f"Ask {workflow.settings.assistant_name}: ")
    )

    try:
        # Single dispatcher loop, single consumer: user turns and
        # background scene events are handled one at a time (no two LLM
        # calls racing each other), but neither producer blocks on the
        # other — a scene event can be queued while a user turn's tool
        # loop is still running, and vice versa. This replaces the old
        # `while True: input()` loop, which had no way to notice anything
        # happening between two explicit turns.
        while camera_manager.is_streaming:
            item = await event_queue.get()
            if item is None:
                break
            if isinstance(item, PerceptionEvent):
                continue
            else:
                await workflow.handle_turn_async(item)
    except KeyboardInterrupt:
        pass
    finally:
        stdin_task.cancel()
        # camera_manager.stop() joins background threads — run off the
        # event loop so shutdown doesn't block it.
        await asyncio.to_thread(camera_manager.stop)


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()