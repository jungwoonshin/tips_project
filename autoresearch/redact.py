"""Oracle-blindness layer for the meta-research agent.

Consumes output/owl_counterfactual_v4/answers/<tid>.json and produces
redacted snapshots that strip:
  - ground_truth / oracle_answer
  - agent_original_final_answer (often contains a near-copy of the oracle)

Keeps:
  - trajectory_id, outcome, published_sufficient_set, fixes (node_id/type/field/content)
  - per-seed replay final_answer strings + flipped booleans
  - error-step rationales and oracle-blind diagnostic hints

A unit test asserts no oracle substring leaks into the redacted output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from autoresearch import REPO_ROOT

_ANSWERS_DIR = REPO_ROOT / "output" / "owl_counterfactual_v4" / "answers"

_REDACTED_SENTINEL = "<REDACTED>"
_REDACTED_FIELDS = {"ground_truth", "oracle_answer", "agent_original_final_answer"}


def _scrub_oracle(text: str, oracle: str) -> str:
    """Replace every word-boundary occurrence of oracle in text with sentinel.
    Kept as a belt-and-suspenders guard on replay final_answer strings (some
    replays might echo the oracle directly — we keep that echo too valuable to
    drop entirely, but neuter any literal oracle substring)."""
    if not oracle or len(oracle) < 3:
        return text  # too short to scrub safely
    return re.sub(r"\b" + re.escape(oracle) + r"\b", _REDACTED_SENTINEL, text or "",
                  flags=re.IGNORECASE)


def redact_answer(answer: dict) -> dict:
    """Return a copy of an `answers/<tid>.json` payload with oracle removed."""
    oracle = answer.get("ground_truth") or ""
    out = {k: v for k, v in answer.items() if k not in _REDACTED_FIELDS}
    out["ground_truth"] = _REDACTED_SENTINEL
    out["agent_original_final_answer"] = _REDACTED_SENTINEL

    # Scrub replay final_answer strings (belt-and-suspenders — strict-match flip
    # strings would literally contain the oracle).
    cleaned = []
    for r in out.get("replay_results", []) or []:
        cr = dict(r)
        cr["final_answer"] = _scrub_oracle(cr.get("final_answer") or "", oracle)
        cleaned.append(cr)
    if cleaned:
        out["replay_results"] = cleaned
    return out


def build_dev_snapshot(trajectory_ids: Iterable[str]) -> list[dict]:
    """Build a list of redacted per-trajectory snapshots for agent context."""
    snapshot = []
    for tid in trajectory_ids:
        path = _ANSWERS_DIR / f"{tid}.json"
        if not path.exists():
            # Missing answer = trajectory never made it through pipeline; skip.
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        snapshot.append(redact_answer(payload))
    return snapshot


def assert_no_oracle_leak(redacted: dict, original: dict) -> None:
    """Raise AssertionError if the redacted payload contains a substring
    of the original oracle. For use in unit tests."""
    oracle = (original.get("ground_truth") or "").strip()
    if not oracle or len(oracle) < 3:
        return
    dumped = json.dumps(redacted, ensure_ascii=False).lower()
    if re.search(r"\b" + re.escape(oracle.lower()) + r"\b", dumped):
        raise AssertionError(
            f"oracle '{oracle}' leaked into redacted payload for "
            f"{original.get('trajectory_id')}"
        )
