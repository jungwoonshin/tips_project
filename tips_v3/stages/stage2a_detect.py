"""Stage 2a: Sonnet 4.5 detection with k-sample agreement confidence."""

from __future__ import annotations

import logging
from collections import Counter

from tips_v3.config import (
    DETECTION_K_SAMPLES,
    DETECTION_TEMPERATURE,
    ERROR_TYPES,
    N_MAX_CANDIDATES,
    TAU_DROP,
)
from tips_v3 import checkpoint
from tips_v3.io.schema import Candidate, Trajectory
from tips_v3.llm.prompts import detect_2a
from tips_v3.llm.sonnet_client import SonnetClient, parse_json

log = logging.getLogger(__name__)

STAGE = "stage2a"


def _parse_sample(text: str) -> list[dict]:
    try:
        data = parse_json(text)
    except Exception as exc:
        log.warning("stage2a parse failure: %s", exc)
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("candidates", "items", "results", "errors"):
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


def _hint_leaks_oracle(hint: str, oracle: str) -> bool:
    """Cheap leakage filter on diagnostic_hint.

    Uses word-boundary matching so short/common oracles ('41', 'yes') don't
    cause spurious drops; still catches literal oracle phrases.
    """
    import re
    oracle = (oracle or "").strip().lower()
    if not oracle or len(oracle) < 3:
        return False
    pat = r"\b" + re.escape(oracle) + r"\b"
    return re.search(pat, (hint or "").lower()) is not None


def detect(traj: Trajectory, client: SonnetClient) -> list[Candidate]:
    cached = checkpoint.load(traj.trajectory_id, STAGE)
    if cached is not None:
        return [Candidate(**c) for c in cached]

    msg = detect_2a.build(traj)
    samples = client.k_samples(
        msg,
        k=DETECTION_K_SAMPLES,
        temperature=DETECTION_TEMPERATURE,
        max_tokens=4096,
    )

    counts: Counter[str] = Counter()
    first_seen: dict[str, dict] = {}
    for sample_text in samples:
        for entry in _parse_sample(sample_text):
            nid = str(entry.get("node_id") or "")
            if not nid:
                continue
            ptype = entry.get("predicted_type")
            if ptype not in ERROR_TYPES:
                continue
            counts[nid] += 1
            first_seen.setdefault(nid, entry)

    candidates: list[Candidate] = []
    valid_ids = {n.step_id for n in traj.nodes}
    for nid, cnt in counts.most_common():
        if nid not in valid_ids:
            continue
        conf = cnt / DETECTION_K_SAMPLES
        if conf < TAU_DROP:
            continue
        raw = first_seen[nid]
        hint = raw.get("diagnostic_hint") or ""
        if _hint_leaks_oracle(hint, traj.oracle_answer):
            log.warning("dropping candidate %s: hint leaks oracle", nid)
            continue
        candidates.append(
            Candidate(
                node_id=nid,
                level=raw.get("level") or "worker",
                role=raw.get("role") or "",
                predicted_type=raw.get("predicted_type"),
                confidence=conf,
                rationale=raw.get("rationale") or "",
                diagnostic_hint=hint,
            )
        )
        if len(candidates) >= N_MAX_CANDIDATES:
            break

    checkpoint.save(traj.trajectory_id, STAGE, [c.__dict__ for c in candidates])
    log.info("stage2a %s: %d candidates", traj.trajectory_id, len(candidates))
    return candidates
