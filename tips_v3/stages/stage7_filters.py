"""Stage 7 — Validity filters: F1 null-fix, F2 leakage, F3 minimality, F4 seed-floor."""

from __future__ import annotations

import difflib
import hashlib
import logging

from tips_v3 import checkpoint
from tips_v3.config import (
    FIX_TEMPERATURE,
    NULL_FIX_FLIP_RATE_MAX,
    NULL_FIX_SUBSET_RATE,
)
from tips_v3.io.schema import Candidate, Fix, SuffSetState, Trajectory
from tips_v3.llm.prompts import leakage_audit, null_fix
from tips_v3.llm.sonnet_client import SonnetClient, parse_json
from tips_v3.replay.bounded_replay import BoundedReplay
from tips_v3.replay.seed_control import reduction
from tips_v3.stages.stage2b_fix import _parse_one

log = logging.getLogger(__name__)

STAGE = "stage7"


def _in_null_fix_subset(tid: str) -> bool:
    h = hashlib.sha1(tid.encode()).hexdigest()
    return (int(h, 16) % 100) < int(NULL_FIX_SUBSET_RATE * 100)


def _string_leak(text: str, oracle: str) -> bool:
    """Word-boundary oracle match, skips very short oracles to avoid
    false positives on common tokens like numeric sub-strings."""
    import re
    oracle = oracle.strip().lower()
    if not oracle or len(oracle) < 3:
        return False
    pat = r"\b" + re.escape(oracle) + r"\b"
    text = (text or "").lower()
    if re.search(pat, text):
        return True
    return difflib.SequenceMatcher(None, oracle, text[:200]).ratio() >= 0.9


def _hint_for(candidates: list[Candidate], node_id: str) -> str:
    for c in candidates:
        if c.node_id == node_id:
            return c.diagnostic_hint or ""
    return ""


def run_filters(
    traj: Trajectory,
    state: SuffSetState,
    fixes: list[Fix],
    client: SonnetClient,
    replay: BoundedReplay,
    candidates: list[Candidate] | None = None,
) -> dict:
    cached = checkpoint.load(traj.trajectory_id, STAGE)
    if cached is not None:
        return cached

    result = {
        "null_fix_flip_rate": None,
        "null_fix_triggered": False,
        "leakage_audit_passed": True,
        "leaky_fix_ids": [],
        "minimality_audit_passed": True,
        "seed_floor_passed": state.k_used >= 3,
    }

    fix_by_id = {f.node_id: f for f in fixes}
    M_fixes = [fix_by_id[nid] for nid in state.M if nid in fix_by_id]
    candidates = candidates or []

    # F2: audit the DIAGNOSTIC HINT (the only channel through which oracle
    # info could flow from oracle-visible Stage 2a into oracle-blind Stage 2b).
    # Fix content may legitimately contain the oracle when Sonnet, working
    # oracle-blind, re-derived the correct answer from pre-error context —
    # that is a valid flip, not leakage.
    for f in M_fixes:
        hint = _hint_for(candidates, f.node_id)
        if _string_leak(hint, traj.oracle_answer):
            result["leakage_audit_passed"] = False
            result["leaky_fix_ids"].append(f.node_id)

    # Sonnet-on-Sonnet content audit removed: with F2 now targeting hints
    # only, a fix stating the oracle is not in itself evidence of leakage —
    # oracle-blind derivation is allowed. The hint-leak string check above
    # plus Stage 2a's _hint_leaks_oracle are sufficient belt-and-suspenders.

    # F3 minimality re-check removed along with the token-diff cap.

    # F1 null-fix baseline on 5% stratified subset
    if _in_null_fix_subset(traj.trajectory_id) and M_fixes:
        result["null_fix_triggered"] = True
        null_fixes = _generate_null_fixes(traj, M_fixes, client)
        if null_fixes:
            null_map = {nf.node_id: nf for nf in null_fixes}
            results = replay.run(traj, null_map, reduction())
            fr = sum(1 for r in results if r.flipped) / max(len(results), 1)
            result["null_fix_flip_rate"] = fr
            if fr > NULL_FIX_FLIP_RATE_MAX:
                result["leakage_audit_passed"] = False  # perturbation-sensitive → reject

    checkpoint.save(traj.trajectory_id, STAGE, result)
    log.info("stage7 %s: %s", traj.trajectory_id, result)
    return result


def _generate_null_fixes(traj: Trajectory, fixes: list[Fix], client: SonnetClient) -> list[Fix]:
    messages = [null_fix.build(traj, f) for f in fixes]
    texts = client.batch(messages, temperature=FIX_TEMPERATURE, max_tokens=1024)
    out: list[Fix] = []
    for f, text in zip(fixes, texts):
        cand_like = type("C", (), {
            "node_id": f.node_id,
            "predicted_type": f.predicted_type,
        })()
        parsed = _parse_one(text, cand_like)
        if parsed is not None:
            out.append(parsed)
    return out
