"""Stage 6: k=5 final verification."""

from __future__ import annotations

import logging

from tips_v3 import checkpoint
from tips_v3.config import FINAL_FLIP_THRESHOLD, FINAL_VERIFY_K
from tips_v3.io.schema import Fix, SuffSetState, Trajectory
from tips_v3.replay.bounded_replay import BoundedReplay
from tips_v3.replay.seed_control import final_verify

log = logging.getLogger(__name__)

STAGE = "stage6"


def verify(
    traj: Trajectory,
    state: SuffSetState,
    fixes: list[Fix],
    replay: BoundedReplay,
) -> SuffSetState:
    cached = checkpoint.load(traj.trajectory_id, STAGE)
    if cached is not None:
        return SuffSetState(**cached)

    fix_by_id = {f.node_id: f for f in fixes}
    fix_map = {nid: fix_by_id[nid] for nid in state.M if nid in fix_by_id}

    results = replay.run(traj, fix_map, final_verify())
    flips = sum(1 for r in results if r.flipped)
    flip_rate = flips / FINAL_VERIFY_K
    seeds = {r.seed: ("flip" if r.flipped else ("error" if r.error else "no_flip")) for r in results}

    passed = flips >= FINAL_FLIP_THRESHOLD
    new_state = SuffSetState(
        trajectory_id=traj.trajectory_id,
        stage="verify" if passed else "verify_failed",
        M=list(state.M),
        flip_seeds=seeds,
        flip_rate=flip_rate,
        k_used=FINAL_VERIFY_K,
    )
    checkpoint.save(traj.trajectory_id, STAGE, new_state.__dict__)
    log.info(
        "stage6 %s: flips=%d/%d passed=%s",
        traj.trajectory_id, flips, FINAL_VERIFY_K, passed,
    )
    return new_state


def passed(state: SuffSetState) -> bool:
    return state.stage == "verify" and state.flip_rate >= FINAL_FLIP_THRESHOLD / FINAL_VERIFY_K
