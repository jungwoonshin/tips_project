"""TIPS AutoResearch — autonomous iteration on the TIPS pipeline.

Goal: measure how much flip-rate headroom an autonomous research agent can
extract from prompt + hyperparameter search on the TIPS pipeline, with the
replay harness and oracle-blindness invariants frozen.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_ROOT = Path(__file__).resolve().parent
RUNS_DIR = AUTORESEARCH_ROOT / "runs"
CACHE_DIR = AUTORESEARCH_ROOT / "cache"

# v4/ is the FROZEN baseline dataset (read-only source of per-trajectory
# answers consumed by redact.py). autoresearch writes its own pipeline runs
# to v5/ so v4 stays intact as the comparison point.
FROZEN_BASELINE_DIR = REPO_ROOT / "output" / "owl_counterfactual_v4"
AUTORESEARCH_OUTPUT_DIR = REPO_ROOT / "output" / "owl_counterfactual_v5"

# DEV/TEST split (frozen — editing this invalidates published results)
DEV_TRAJECTORIES = [
    "gaia_validation_0006",   # tool-swap-flippable (canary)
    "gaia_validation_0002",   # terminal-reasoning
    "gaia_validation_0011",   # info-not-reachable
    "gaia_validation_0016",   # prior-lock-in
    "gaia_validation_0028",   # borderline
]
