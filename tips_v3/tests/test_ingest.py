"""Ingest smoke test against real validation_fresh trajectories."""

from __future__ import annotations

from tips_v3.config import INPUT_DIR
from tips_v3.io.ingest import iter_trajectories


def test_ingest_parses_validation_fresh():
    seen = 0
    for traj in iter_trajectories(INPUT_DIR):
        assert traj.trajectory_id
        assert traj.oracle_answer
        assert traj.agent_final_answer
        assert traj.nodes
        for n in traj.nodes:
            assert n.step_id is not None
            assert n.level in {"planner", "worker"}
        seen += 1
        if seen >= 3:
            break
    assert seen > 0, "no trajectories ingested"
