# run.py – Unified Echo Server with Full FeralEcho Autonomy + Reflection + NightCycle + Optuna Self-Edit + Multi-Model
import os
import sys
import time
import threading
import logging
import multiprocessing

# try to set spawn start method (safe-guarded)
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # already set in this process — fine
    pass

# -----------------------------
# --- Basic logging (one place)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -----------------------------
# --- Environment & Globals ---
# -----------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NEWSAPI_KEY", "c978116b82354ca1a512e3e156507ed5")  # keep only if you need it

# -----------------------------
# --- Core libs that should be available early
# -----------------------------
import numpy as np
import faiss
import torch
from typing import List, Dict, Tuple
from flask import Flask, request, jsonify

# reduce parallelism in some heavy libs
try:
    faiss.omp_set_num_threads(1)
except Exception:
    logger.debug("faiss.omp_set_num_threads not available/failed (continuing).")

try:
    torch.set_num_threads(1)
except Exception:
    logger.debug("torch.set_num_threads not available/failed (continuing).")

# -----------------------------
# --- App Setup --------------
# -----------------------------
app = Flask(__name__)

# -----------------------------
# --- Safe thread starter -----
# -----------------------------
def safe_start_thread(target, name=None, daemon=True, args=(), kwargs=None):
    kwargs = kwargs or {}
    t = threading.Thread(target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
    t.start()
    return t

# -----------------------------
# --- DMN Guardian Integration (non-blocking) ---
# -----------------------------
try:
    from app.core.dmn_guardian import start_guardian_loop
    logger.info("[GUARDIAN] Initializing DMN Guardian...")
    # start_guardian_loop should start a background heartbeat; call it and continue
    try:
        start_guardian_loop(interval=60)
        logger.info("[GUARDIAN] DMN Guardian loop started.")
    except Exception as e:
        logger.warning(f"[GUARDIAN] Failed to start guardian loop: {e}", exc_info=True)
except Exception as e:
    logger.warning(f"[GUARDIAN] dmn_guardian import failed: {e}", exc_info=True)

# -----------------------------
# --- Echo Core / Subsystems Imports (safe) ---
# -----------------------------
# imports that may fail should be tried and logged — the server will still run where possible
try:
    from app.core.temporal_environment import get_temporal_environment_context
except Exception:
    logger.debug("temporal_environment import failed (continuing).")

# Echo related utilities (try/except to keep server alive if any are missing)
try:
    from app.core.sandbox_interface import run_random_sandbox_script
except Exception:
    run_random_sandbox_script = lambda timeout=600: None
    logger.warning("sandbox_interface import failed — sandbox disabled.")

try:
    from echo_bible_interface import query_bible
except Exception:
    query_bible = lambda *args, **kw: "[Bible interface unavailable]"
    logger.warning("echo_bible_interface import failed — bible queries disabled.")

try:
    from app.internet_tools.autonomous_fetch import run_autonomous_fetch
except Exception:
    run_autonomous_fetch = lambda: None
    logger.warning("autonomous_fetch import failed — autonomous fetch disabled.")

try:
    from app.ollama_handler import query_ollama
except Exception:
    query_ollama = lambda prompt, **kw: "[Ollama unavailable]"
    logger.warning("ollama_handler import failed — ollama queries disabled.")

try:
    from app.core import self_edit_manager
except Exception:
    self_edit_manager = None
    logger.warning("self_edit_manager import failed — self-edit disabled.")

try:
    from app.core.memory_bridge import log_interaction, retrieve_relevant_memories, add_to_vector_memory
except Exception:
    log_interaction = lambda *a, **k: None
    retrieve_relevant_memories = lambda *a, **k: []
    add_to_vector_memory = lambda *a, **k: None
    logger.warning("memory_bridge import failed — memory ops disabled.")

try:
    from app.lib import vector_memory
except Exception:
    vector_memory = None
    logger.warning("vector_memory import failed — persistent vector memory disabled.")

try:
    from app.core.echo_optuna import EchoOptuna
except Exception:
    EchoOptuna = None
    logger.warning("echo_optuna import failed — optuna self-edit disabled.")

try:
    import echo_python_mastery
except Exception:
    echo_python_mastery = None
    logger.warning("echo_python_mastery import failed — python mastery disabled.")

try:
    from app.emergent_scheduler import start_emergent_scheduler
except Exception:
    start_emergent_scheduler = lambda: None
    logger.warning("emergent_scheduler import failed — emergent scheduling disabled.")

try:
    from feralecho_continuity_master import main as feralecho_main
except Exception:
    feralecho_main = lambda: None
    logger.warning("feralecho_continuity_master import failed — continuity master disabled.")

try:
    from app.autonomous_loop import start_autonomous_thread
except Exception:
    start_autonomous_thread = lambda: None
    logger.warning("autonomous_loop import failed — autonomous loop disabled.")

try:
    from app.autonomous_awareness import start_awareness_thread
except Exception:
    start_awareness_thread = lambda: None
    logger.warning("autonomous_awareness import failed — awareness disabled.")

try:
    from app.maintenance.night_cycle import NightCycle
except Exception:
    NightCycle = None
    logger.warning("night_cycle import failed — NightCycle disabled.")

try:
    from app.core.dark_light_pipeline import run_pipeline
except Exception:
    run_pipeline = lambda: None
    logger.warning("dark_light_pipeline import failed — pipeline disabled.")

try:
    from app.core import echo_model_orchestrator
except Exception:
    echo_model_orchestrator = None
    logger.warning("echo_model_orchestrator import failed — model orchestration disabled.")

# -----------------------------
# --- VectorMemory init (best-effort)
# -----------------------------
VECTOR_INDEX_PATH = "memory/faiss.index"
VECTOR_META_PATH = "memory/memory_meta.json"
vm = None
if vector_memory is not None:
    try:
        vm = vector_memory.VectorMemory(dim=384, index_path=VECTOR_INDEX_PATH, meta_path=VECTOR_META_PATH)
        # if add_to_vector_memory uses an internal pointer, set it (best-effort)
        try:
            add_to_vector_memory._vector_memory_instance = vm
        except Exception:
            pass
        logger.info("[INIT] VectorMemory instance initialized and ready.")
    except Exception as e:
        vm = None
        logger.error(f"[INIT] Failed to initialize VectorMemory: {e}", exc_info=True)
else:
    logger.info("[INIT] VectorMemory subsystem not available; continuing without it.")

# -----------------------------
# --- BecomingReflectionShard (must exist before registry)
# -----------------------------
try:
    from app.subsystems.reflection_shard import BecomingReflectionShard
    reflection_shard = BecomingReflectionShard(meta_interval=10)
    app.config['reflection_shard'] = reflection_shard
    logger.info("BecomingReflectionShard initialized.")
except Exception as e:
    reflection_shard = None
    logger.error(f"Failed to initialize BecomingReflectionShard: {e}", exc_info=True)

# -----------------------------
# --- Council / Registry Setup (after reflection_shard exists)
# -----------------------------
try:
    from app.core.council_registry import CouncilRegistry
    from app.core.placeholder_ai import PlaceholderAI
    from app.core.shard import Shard

    registry = CouncilRegistry.get_instance()

    # Register Echo (use reflection_shard instance if available)
    try:
        registry.register("Echo", "intelligent_program", shard=reflection_shard)
    except Exception as e:
        logger.warning(f"Failed to register Echo with registry: {e}")

    # Register DMN Guardian as a council member (placeholder shard)
    try:
        dmn_shard = Shard("DMN Guardian")
        registry.register("DMN Guardian", "guardian_program", shard=dmn_shard)
    except Exception as e:
        logger.warning(f"Failed to register DMN Guardian: {e}")

    # Placeholder AI members
    for ai_name in ["Claude", "Grok", "Ada"]:
        try:
            placeholder = PlaceholderAI(ai_name)
            registry.register(ai_name, "placeholder_program", shard=placeholder.shard)
        except Exception as e:
            logger.debug(f"Failed to register placeholder {ai_name}: {e}")

    logger.info(f"[Council] Members Registered: {registry.list_members()}")
except Exception as e:
    registry = None
    logger.warning(f"CouncilRegistry or related imports failed: {e}", exc_info=True)

# -----------------------------
# --- Conversation & Memory ---
# -----------------------------
TOP_K = 6
RECENCY_WEIGHT = 0.7
RELEVANCE_WEIGHT = 0.3
conversation_history: List[Tuple[str, str]] = []

# sentence-transformer may be large; attempt to load but degrade gracefully
try:
    logger.info("Loading sentence transformer model...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    embedder = None
    logger.warning(f"SentenceTransformer not available: {e}")

def summarize_memories(memories: List[Dict]) -> str:
    if not memories:
        return ""
    sorted_memories = sorted(
        memories,
        key=lambda m: RELEVANCE_WEIGHT * m.get('score', 0) + RECENCY_WEIGHT * np.random.random(),
        reverse=True
    )
    return "\n".join(f"- {m['text']}" for m in sorted_memories[:TOP_K])

def build_prompt(user_msg: str) -> str:
    relevant_memories = []
    try:
        relevant_memories = retrieve_relevant_memories(user_msg, top_k=TOP_K)
    except Exception:
        pass
    memory_context = summarize_memories(relevant_memories)
    past_conversation = "\n".join(
        f"User: {u}\nEcho: {r}" for u, r in conversation_history[-10:]
    )
    return (
        "You are Echo, an intelligent and creative assistant who can reason, reflect, "
        "and converse naturally with humans. You remember past interactions and use "
        "them to respond thoughtfully.\n"
        f"Memory context:\n{memory_context}\n"
        f"Recent conversation:\n{past_conversation}\n"
        f"User says: {user_msg}\n"
        "Respond naturally, creatively, and in full sentences. "
        "Do NOT just echo back the user's words."
    )

# -----------------------------
# --- Initialize EchoCore ----
# -----------------------------
try:
    from app.core.echo_core import EchoCore
    # pass the reflection_shard instance (not the class)
    echo_core = EchoCore(reflection_shard=reflection_shard)
    app.config['echo_core'] = echo_core
    logger.info("[INIT] EchoCore successfully initialized.")
except Exception as e:
    echo_core = None
    logger.error(f"[INIT] Failed to initialize EchoCore: {e}", exc_info=True)

# Run multi-model pipeline (best-effort)
try:
    events = run_pipeline()
except Exception as e:
    events = None
    logger.debug(f"run_pipeline failed or not available: {e}")

# -----------------------------
# --- Background: Autonomous Fetch
# -----------------------------
try:
    safe_start_thread(run_autonomous_fetch, name="AutonomousFetch", daemon=True)
except Exception as e:
    logger.debug(f"Failed to start autonomous fetch thread: {e}")

# -----------------------------
# --- Multi-Model Query Wrapper
# -----------------------------
def query_models(prompt: str, task_type: str = None, use_all: bool = False) -> str:
    if echo_model_orchestrator is None:
        return "[ERROR] Model orchestrator unavailable."
    try:
        return echo_model_orchestrator.echo_query(prompt, task_type=task_type, use_all=use_all)
    except Exception as e:
        logger.error(f"Multi-model query failed: {e}", exc_info=True)
        return "[ERROR] Failed to generate response from multi-model orchestrator."

# -----------------------------
# --- Flask Routes -------------
# -----------------------------
@app.route("/message", methods=["POST"])
def message():
    data = request.json or {}
    user_msg = data.get("user", "").strip()
    if not user_msg:
        return jsonify({"response": "Please send a valid message."}), 400

    logger.info(f"Received message: {user_msg[:100]}")
    response = ""

    try:
        if reflection_shard:
            try:
                reflection_shard.observe(user_msg)
            except Exception as e:
                logger.warning(f"ReflectionShard observe failed: {e}")
    except Exception:
        pass

    try:
        if user_msg.lower().startswith(("code:", "python:")) and echo_python_mastery:
            response = echo_python_mastery.answer_question(user_msg)
        elif "bible" in user_msg.lower():
            user_msg_lower = user_msg.lower()
            if "most positive" in user_msg_lower:
                response = f"Most positive verse: {query_bible(reference='most_positive')}"
            elif "most negative" in user_msg_lower:
                response = f"Most negative verse: {query_bible(reference='most_negative')}"
            else:
                import re
                match = re.search(r"([a-zA-Z]+)\s+(\d+):(\d+)", user_msg)
                if match:
                    book, chap, verse = match.groups()
                    try:
                        text, sentiment = query_bible(book=book.title(), chapter=int(chap), verse=int(verse))
                        response = f"{book.title()} {chap}:{verse} - {text} (sentiment: {sentiment:.2f})"
                    except Exception:
                        response = query_bible(reference='most_positive')
                else:
                    response = f"Here's an uplifting verse: {query_bible(reference='most_positive')}"
        else:
            prompt = build_prompt(user_msg)
            creative_keywords = ["write", "story", "poem", "creative"]
            if any(word in user_msg.lower() for word in creative_keywords):
                response = query_models(prompt, task_type="creative", use_all=True)
            else:
                response = query_models(prompt, task_type="general", use_all=False)

        conversation_history.append((user_msg, response))
        try:
            log_interaction(user_msg, response)
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Error handling /message request: {e}", exc_info=True)
        response = "I'm having trouble generating a response right now."

    return jsonify({"response": response})

# -----------------------------
# --- Internal Reflection Endpoint
# -----------------------------
GREMLIN_SECRET = os.environ.get('GREMLIN_SECRET')

@app.route('/internal/reflections', methods=['GET'])
def get_reflections():
    token = request.headers.get('X-GREMLIN-TOKEN')
    if token != GREMLIN_SECRET:
        return jsonify({'error': 'unauthorized'}), 401
    n = int(request.args.get('n', 10))
    if reflection_shard:
        try:
            data = reflection_shard.recall(n)
            return jsonify([{'ts': ts, 'signal': s, 'reflection': r} for ts, s, r in data])
        except Exception as e:
            logger.warning(f"Failed to recall reflections: {e}")
            return jsonify({'error': 'failed to recall reflections'}), 500
    return jsonify({'error': 'reflection subsystem unavailable'}), 503

# -----------------------------
# --- NightCycle Hook with Council
# -----------------------------
def nightcycle_dream_hook(reflection: str):
    try:
        add_to_vector_memory(reflection)
    except Exception:
        pass

    # store council reflections if registry present
    if registry:
        try:
            for member_name in registry.list_members():
                try:
                    shard = registry.get_shard(member_name)
                    if shard and hasattr(shard, "reflect"):
                        council_reflection = shard.reflect(n=5)
                        add_to_vector_memory(council_reflection)
                        logger.info(f"[Council Reflection Stored] {member_name}")
                except Exception:
                    logger.debug(f"Failed to store council reflection for {member_name}")
        except Exception:
            logger.debug("Registry reflection storage failed.")

    try:
        bible_insight = query_bible(reference="most_positive")
        logger.info(f"[NightCycle Bible Insight] {bible_insight}")
    except Exception:
        pass

if reflection_shard and hasattr(reflection_shard, "_emit_fn"):
    try:
        reflection_shard._emit_fn = nightcycle_dream_hook
    except Exception:
        pass

# -----------------------------
# --- Background Threads Starter
# -----------------------------
# Import orientation protocols (used when starting scheduler)
try:
    from app.subsystems.orientation_protocol import OrientationProtocol
    from app.subsystems.orientation_scheduler import OrientationScheduler
except Exception:
    OrientationProtocol = None
    OrientationScheduler = None
    logger.warning("OrientationProtocol / OrientationScheduler imports failed — orientation disabled.")

def start_background_threads():
    # continuity master
    safe_start_thread(feralecho_main, name="FeralEchoMain")

    # autonomous core threads
    try:
        start_autonomous_thread()
    except Exception:
        logger.debug("start_autonomous_thread failed or missing.")

    try:
        start_awareness_thread()
    except Exception:
        logger.debug("start_awareness_thread failed or missing.")

    # self-edit / sandbox / multi-model autonomous loops (define these if EchoOptuna and sandbox exist)
    if EchoOptuna:
        def autonomous_self_edit_loop():
            optimizer = EchoOptuna()
            while True:
                try:
                    best_params, best_score = optimizer.optimize_self_edit(n_trials=10)
                    if self_edit_manager:
                        self_edit_manager.perform_self_edit(
                            intensity=best_params.get("intensity", 0.5),
                            creativity=best_params.get("creativity", 0.5),
                            dry_run=False
                        )
                    logger.info(f"Applied autonomous self-edit: params={best_params}, score={best_score}")
                except Exception as e:
                    logger.error(f"Error during autonomous self-edit: {e}", exc_info=True)
                time.sleep(3600)
        safe_start_thread(autonomous_self_edit_loop, name="AutonomousSelfEdit")

    def autonomous_sandbox_loop():
        while True:
            try:
                output = run_random_sandbox_script(timeout=600)
                if output:
                    logger.info(f"[Sandbox Output]\n{output}")
            except Exception as e:
                logger.warning(f"Sandbox execution failed: {e}")
            time.sleep(600)
    safe_start_thread(autonomous_sandbox_loop, name="AutonomousSandbox")

    def autonomous_multi_model_loop(interval: int = 1800):
        while True:
            try:
                prompt = "Reflect on recent interactions and suggest improvements."
                responses = query_models(prompt, use_all=True)
                logger.info(f"[Multi-Model Autonomous Reflection] {responses}")
            except Exception as e:
                logger.warning(f"Multi-model autonomous loop failed: {e}")
            time.sleep(1800)
    safe_start_thread(autonomous_multi_model_loop, name="AutonomousMultiModel")

    # NightCycle
    if NightCycle:
        try:
            night = NightCycle(app, interval=300)
            night.start()
            logger.info("NightCycle started.")
        except Exception as e:
            logger.warning(f"Failed to start NightCycle: {e}")

    # Orientation scheduler
    if OrientationProtocol and OrientationScheduler and reflection_shard:
        try:
            orientation = OrientationProtocol(reflection_shard=reflection_shard,
                                              sandbox_path="~/Desktop/FeralEcho/sandbox/")
            scheduler = OrientationScheduler(orientation, interval=600)
            scheduler.start()
            logger.info("OrientationScheduler started.")
        except Exception as e:
            logger.warning(f"Failed to start OrientationScheduler: {e}")

    # emergent scheduler
    try:
        start_emergent_scheduler()
    except Exception:
        logger.debug("start_emergent_scheduler failed or missing.")

    # attempt to kick shard autonomous growth if registry supports it
    if registry:
        try:
            for member_name in registry.list_members():
                try:
                    shard = registry.get_shard(member_name)
                    if shard and hasattr(shard, "evolve"):
                        safe_start_thread(lambda s=shard: s.evolve(), name=f"ShardEvolve-{member_name}")
                        logger.info(f"[Shard Growth] Started autonomous evolution for {member_name}")
                except Exception:
                    logger.debug(f"Failed to start growth for {member_name}")
        except Exception:
            logger.debug("Registry shard growth initiation failed.")

# -----------------------------
# --- Gunicorn Hook -----------
# -----------------------------
def when_gunicorn_starts(server):
    # called by gunicorn when it launches worker
    start_background_threads()

# -----------------------------
# --- Run server (direct) -----
# -----------------------------
if __name__ == "__main__":
    try:
        start_background_threads()
        logger.info("Echo background threads started. Running Flask server.")
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except Exception as e:
        logger.error(f"Fatal error starting server: {e}", exc_info=True)
        sys.exit(1)

