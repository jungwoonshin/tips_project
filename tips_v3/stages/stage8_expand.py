"""Stage 8: adversarial expansion pass for trajectories that failed Stage 4/6."""

from __future__ import annotations

import logging
from collections import Counter

from tips_v3 import checkpoint
from tips_v3.config import (
    DETECTION_K_SAMPLES,
    DETECTION_TEMPERATURE,
    ERROR_TYPES,
    N_MAX_CANDIDATES,
    TAU_DROP,
)
from tips_v3.io.schema import Candidate, Trajectory
from tips_v3.llm.prompts import expand
from tips_v3.llm.sonnet_client import SonnetClient, parse_json

log = logging.getLogger(__name__)

STAGE = "stage8"


def expand_candidates(
    traj: Trajectory,
    prior: list[Candidate],
    client: SonnetClient,
) -> list[Candidate]:
    cached = checkpoint.load(traj.trajectory_id, STAGE)
    if cached is not None:
        return [Candidate(**c) for c in cached]

    msg = expand.build(traj, prior)
    samples = client.k_samples(
        msg,
        k=DETECTION_K_SAMPLES,
        temperature=DETECTION_TEMPERATURE,
        max_tokens=4096,
    )

    prior_ids = {c.node_id for c in prior}
    counts: Counter[str] = Counter()
    first: dict[str, dict] = {}
    valid_ids = {n.step_id for n in traj.nodes}

    for text in samples:
        try:
            data = parse_json(text)
        except Exception:
            continue
        entries: list[dict] = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            for k in ("candidates", "items", "results", "errors"):
                v = data.get(k)
                if isinstance(v, list):
                    entries = v
                    break
        for entry in entries:
            nid = str(entry.get("node_id") or "")
            if nid in prior_ids or nid not in valid_ids:
                continue
            if entry.get("predicted_type") not in ERROR_TYPES:
                continue
            counts[nid] += 1
            first.setdefault(nid, entry)

    new_cands: list[Candidate] = []
    for nid, cnt in counts.most_common():
        conf = cnt / DETECTION_K_SAMPLES
        if conf < TAU_DROP:
            continue
        e = first[nid]
        new_cands.append(
            Candidate(
                node_id=nid,
                level=e.get("level") or "worker",
                role=e.get("role") or "",
                predicted_type=e.get("predicted_type"),
                confidence=conf,
                rationale=e.get("rationale") or "",
                diagnostic_hint=e.get("diagnostic_hint") or "",
            )
        )
        if len(new_cands) >= N_MAX_CANDIDATES:
            break

    checkpoint.save(traj.trajectory_id, STAGE, [c.__dict__ for c in new_cands])
    log.info("stage8 %s: %d new candidates", traj.trajectory_id, len(new_cands))
    return new_cands
