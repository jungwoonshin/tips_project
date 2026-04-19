"""Editable-surface allowlist for the meta-research agent.

A patch is valid iff every file it touches is in ALLOWED_FILES and every
config symbol it modifies (in tips_v3/config.py only) is in ALLOWED_CONFIG_VARS.
Anything else → reject and record 'invalid_patch' in history.
"""

from __future__ import annotations

from pathlib import Path

from autoresearch import REPO_ROOT


ALLOWED_FILES: set[str] = {
    "tips_v3/llm/prompts/detect_2a.py",
    "tips_v3/llm/prompts/fix_2b.py",
    "tips_v3/llm/prompts/expand.py",
    "tips_v3/llm/prompts/null_fix.py",
    "tips_v3/config.py",
}

# Config symbols the agent may modify (only inside tips_v3/config.py).
ALLOWED_CONFIG_VARS: set[str] = {
    "DETECTION_TEMPERATURE",
    "FIX_TEMPERATURE",
    "FIX_SAMPLES_PER_NODE",
    "DETECTION_K_SAMPLES",
    "TAU_DROP",
    "UNCERTAIN_LOWER",
    "UNCERTAIN_UPPER",
    "N_MAX_CANDIDATES",
    "M_CAP",
    "ADAPTIVE_K_INITIAL",
    "ADAPTIVE_K_TIEBREAK",
    "REDUCTION_K",
    "GREEDY_STEP",
    "REDUCTION_STEP",
    "TYPE_PRIORITY",
    "TYPE_ALLOWED_FIELDS",
    "FINAL_FLIP_THRESHOLD",
}

# Must NOT be edited even inside allowed files (structural invariants):
FORBIDDEN_SYMBOLS: set[str] = {
    "SEEDS_GREEDY_INITIAL", "SEEDS_GREEDY_TIEBREAK",
    "SEEDS_REDUCTION", "SEEDS_FINAL_VERIFY",
    "SONNET_MODEL_ID", "GEMMA_MODEL_ID", "FINAL_VERIFY_K",
    "OPENROUTER_BASE_URL", "OPENROUTER_API_KEY_ENV",
    "assert_oracle_absent",  # in fix_2b.py
    "gaia_scorer",            # in tips_v3/eval.py (not allowlisted anyway)
}


def path_is_allowed(path: str | Path) -> bool:
    """Check whether a relative path (from repo root) is in the allowlist."""
    p = str(path).replace("\\", "/").lstrip("./")
    if p.startswith(str(REPO_ROOT) + "/"):
        p = p[len(str(REPO_ROOT)) + 1:]
    return p in ALLOWED_FILES


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()
