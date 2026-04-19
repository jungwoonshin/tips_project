"""Stage 5: reverse-order reduction pass over M."""

from __future__ import annotations

import logging

from tips_v3 import checkpoint
from tips_v3.config import REDUCTION_STEP
from tips_v3.io.schema import Fix, SuffSetState, Trajectory
from tips_v3.replay.bounded_replay import BoundedReplay
from tips_v3.replay.seed_control import reduction

log = logging.getLogger(__name__)

STAGE = "stage5"


def reduce(
    traj: Trajectory,
    state: SuffSetState,
    fixes: list[Fix],
    replay: BoundedReplay,
) -> SuffSetState:
    cached = checkpoint.load(traj.trajectory_id, STAGE)
    if cached is not None:
        return SuffSetState(**cached)

    fix_by_id = {f.node_id: f for f in fixes}
    M = list(state.M)

    # Batched drop: try removing REDUCTION_STEP nodes at a time in reverse
    # insertion order. If the chunked drop still flips, accept and move on;
    # otherwise fall back to single-node drops within that chunk so we don't
    # miss the element that was actually redundant.
    remaining = list(reversed(state.M))
    while remaining:
        chunk = remaining[:REDUCTION_STEP]
        remaining = remaining[REDUCTION_STEP:]
        if len(M) - len(chunk) < 1:
            chunk = chunk[:len(M) - 1]
            if not chunk:
                break
        M_prime = [x for x in M if x not in chunk]
        fix_map = {x: fix_by_id[x] for x in M_prime}
        results = replay.run(traj, fix_map, reduction())
        if sum(1 for r in results if r.flipped) >= 2:
            M = M_prime
            log.info("stage5 %s: dropped chunk %s", traj.trajectory_id, chunk)
            continue
        # Chunk as a whole couldn't be dropped. Fall back to per-node checks
        # so we still find a locally-minimal M.
        for nid in chunk:
            if len(M) <= 1:
                break
            M_prime = [x for x in M if x != nid]
            fix_map = {x: fix_by_id[x] for x in M_prime}
            results = replay.run(traj, fix_map, reduction())
            if sum(1 for r in results if r.flipped) >= 2:
                M = M_prime
                log.info("stage5 %s: dropped node %s (per-node fallback)",
                         traj.trajectory_id, nid)

    fix_map = {x: fix_by_id[x] for x in M}
    results = replay.run(traj, fix_map, reduction())
    flip_rate = sum(1 for r in results if r.flipped) / max(len(results), 1)
    seeds = {r.seed: ("flip" if r.flipped else ("error" if r.error else "no_flip")) for r in results}
    new_state = SuffSetState(
        trajectory_id=traj.trajectory_id,
        stage="reduce",
        M=M,
        flip_seeds=seeds,
        flip_rate=flip_rate,
        k_used=len(results),
    )

    checkpoint.save(traj.trajectory_id, STAGE, new_state.__dict__)
    log.info(
        "stage5 %s: |M|=%d flip_rate=%.2f",
        traj.trajectory_id, len(M), flip_rate,
    )
    return new_state
