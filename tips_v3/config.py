"""Single source of truth for paths, model IDs, thresholds."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = REPO_ROOT / "results" / "owl" / "validation_fresh"
OUTPUT_DIR = REPO_ROOT / "output" / "owl_counterfactual_v4"
CKPT_DIR = OUTPUT_DIR / "_ckpt"
DIFFICULT_DIR = OUTPUT_DIR / "_difficult"
ANSWERS_DIR = OUTPUT_DIR / "answers"
REPLAY_CACHE_DB = OUTPUT_DIR / "_replay_cache.sqlite"
RUN_LOG = OUTPUT_DIR / "run.log"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"

SONNET_MODEL_ID = "anthropic/claude-sonnet-4.5"
GEMMA_MODEL_ID = "google/gemma-4-31b-it"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

DETECTION_TEMPERATURE = 0.7
FIX_TEMPERATURE = 0.5
FIX_SAMPLES_PER_NODE = 3
DETECTION_K_SAMPLES = 3

TAU_DROP = 0.3
UNCERTAIN_LOWER = 0.3
UNCERTAIN_UPPER = 0.6
N_MAX_CANDIDATES = 20
M_CAP = 20

ADAPTIVE_K_INITIAL = 2
ADAPTIVE_K_TIEBREAK = 3
REDUCTION_K = 3
FINAL_VERIFY_K = 5
FINAL_FLIP_THRESHOLD = 3  # >= 3/5 flips required to publish

# Batch-size for greedy growth / reduction shrink. Adds/drops this many nodes
# per step to cut total reruns ~3× vs single-step.
GREEDY_STEP = 3
REDUCTION_STEP = 3

PLAUSIBILITY_SUBSET_RATE = 0.10
NULL_FIX_SUBSET_RATE = 0.05
NULL_FIX_FLIP_RATE_MAX = 0.33

TOKEN_DIFF_CAPS: dict[str, int] = {}  # removed: no cap on fix size

TYPE_PRIORITY = {
    "SEARCH": 0,
    "TOOL": 1,
    "INFORMATION": 2,
    "PLANNING": 3,
    "PREMATURE_TERMINATION": 4,
    "REASONING": 5,
}

TYPE_ALLOWED_FIELDS = {
    "REASONING": {"reasoning"},
    # SEARCH is allowed to swap the search backend (tavily/duckduckgo/wiki) as
    # well as rewrite the query — v2 showed that tool swaps are where most
    # flips actually come from.
    "SEARCH": {"search_query", "tool_name", "tool_args"},
    "TOOL": {"tool_name", "tool_args"},
    # INFORMATION fixes CAN NOT fabricate tool results — if the error is that
    # the retrieval was wrong, the fix should re-shape the tool call (SEARCH
    # or TOOL type) so the replay actually executes a different call and
    # observes a real result.
    "INFORMATION": {"citation", "reasoning"},
    "PLANNING": {"plan"},
    "PREMATURE_TERMINATION": {"continuation"},
}

ERROR_TYPES = tuple(TYPE_ALLOWED_FIELDS.keys())

BATCH_SIZE_TRAJECTORIES = 50
RERUN_CONCURRENCY = 32


def openrouter_api_key() -> str:
    key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{OPENROUTER_API_KEY_ENV} is not set; required for Sonnet 4.5 calls via OpenRouter"
        )
    return key


def ensure_output_dirs() -> None:
    for d in (OUTPUT_DIR, CKPT_DIR, DIFFICULT_DIR, ANSWERS_DIR):
        d.mkdir(parents=True, exist_ok=True)
