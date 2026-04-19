"""Stage 4: backwards elimination from |M|=|C| to a locally minimal M.

Algorithm:
  1. Start with M = all candidates (largest possible repair set).
  2. Verify the full set flips the answer. If not → trajectory is
     structurally unflippable under this candidate set; fail.
  3. Drop one node at a time (lowest priority first). If flipping persists,
     the dropped node was redundant; keep dropping. Otherwise restore it.
  4. Return the locally minimal M — every remaining node is provably
     load-bearing in the sense that removing it breaks the flip.

This starts from maximum fix power and shrinks toward minimality, rather than
the old forward-selection approach which built M up from ∅ and risked
stopping early with a non-minimal or non-flipping set.
"""

from __future__ import annotations

import logging

from tips_v3 import checkpoint
from tips_v3.config import TYPE_PRIORITY
from tips_v3.io.schema import Candidate, Fix, SuffSetState, Trajectory
from tips_v3.replay.bounded_replay import BoundedReplay
from tips_v3.replay.seed_control import greedy_initial, greedy_tiebreak

log = logging.getLogger(__name__)

STAGE = "stage4"

FLIP_ACCEPT = 2  # ≥ 2/3 initial seeds (or ≥ 2/(3+1) after tiebreak)


def _flips_ok(traj, fix_map, replay, state_label=""):
    """Return (ok: bool, results: list[ReplayResult]) after adaptive-k replay."""
    results = replay.run(traj, fix_map, greedy_initial())
    flips = sum(1 for r in results if r.flipped)
    if flips >= FLIP_ACCEPT:
        return True, results
    if flips == 0:
        return False, results
    # 1/3: run tiebreak
    tiebreak = replay.run(traj, fix_map, greedy_tiebreak())
    combined = results + tiebreak
    ok = sum(1 for r in combined if r.flipped) >= FLIP_ACCEPT
    return ok, combined


def construct(
    traj: Trajectory,
    candidates: list[Candidate],
    fixes: list[Fix],
    replay: BoundedReplay,
) -> SuffSetState:
    cached = checkpoint.load(traj.trajectory_id, STAGE)
    if cached is not None:
        return SuffSetState(**cached)

    fix_by_id = {f.node_id: f for f in fixes}
    # Order by (type priority, -confidence). During backwards elimination we
    # drop LOWEST priority first (reversed order) so high-priority nodes —
    # SEARCH/TOOL fixes — are tested for load-bearing-ness last.
    ordered = sorted(
        [c for c in candidates if c.node_id in fix_by_id],
        key=lambda c: (TYPE_PRIORITY.get(c.predicted_type, 99), -c.confidence),
    )

    if not ordered:
        state = SuffSetState(
            trajectory_id=traj.trajectory_id,
            stage="greedy_failed",
            M=[],
            flip_seeds={},
            flip_rate=0.0,
            k_used=0,
        )
        checkpoint.save(traj.trajectory_id, STAGE, state.__dict__)
        return state

    M = [c.node_id for c in ordered]

    # Step 1: verify maximum-fix power actually flips.
    fix_map = {nid: fix_by_id[nid] for nid in M}
    full_ok, full_results = _flips_ok(traj, fix_map, replay, "|C|")
    if not full_ok:
        log.info("stage4 %s: full M=%s does not flip — unflippable", traj.trajectory_id, M)
        state = _state(traj, M, full_results, "greedy_failed")
        checkpoint.save(traj.trajectory_id, STAGE, state.__dict__)
        return state

    log.info("stage4 %s: full M=%s flips; starting backwards elimination",
             traj.trajectory_id, M)

    # Step 2: drop one at a time, LOWEST priority first.
    last_results = full_results
    for cand in reversed(ordered):  # lowest priority first
        if len(M) <= 1:
            break
        if cand.node_id not in M:
            continue  # already dropped
        M_prime = [x for x in M if x != cand.node_id]
        fix_map_prime = {nid: fix_by_id[nid] for nid in M_prime}
        ok, results = _flips_ok(traj, fix_map_prime, replay, f"drop {cand.node_id}")
        if ok:
            M = M_prime
            last_results = results
            log.info("stage4 %s: dropped %s (redundant); |M|=%d",
                     traj.trajectory_id, cand.node_id, len(M))
        else:
            log.info("stage4 %s: kept %s (load-bearing)",
                     traj.trajectory_id, cand.node_id)

    final_state = _state(traj, M, last_results, "greedy")
    checkpoint.save(traj.trajectory_id, STAGE, final_state.__dict__)
    log.info("stage4 %s: final |M|=%d flip_rate=%.2f",
             traj.trajectory_id, len(M), final_state.flip_rate)
    return final_state


def _state(traj: Trajectory, M: list[str], results, stage: str) -> SuffSetState:
    seeds = {r.seed: ("flip" if r.flipped else ("error" if r.error else "no_flip")) for r in results}
    flip_rate = sum(1 for r in results if r.flipped) / max(len(results), 1)
    return SuffSetState(
        trajectory_id=traj.trajectory_id,
        stage=stage,
        M=list(M),
        flip_seeds=seeds,
        flip_rate=flip_rate,
        k_used=len(results),
    )
