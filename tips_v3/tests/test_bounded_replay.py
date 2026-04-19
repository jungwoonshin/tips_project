"""Bounded replay determinism + flip detection against a stub backend."""

from __future__ import annotations

from tips_v3.io.schema import Fix, Node, ReplayResult, Trajectory
from tips_v3.replay.bounded_replay import BoundedReplay, ReplayCache


class _StubBackend:
    """Flips iff fix_map contains `s1` with new_content == 'correct'."""
    def __init__(self):
        self.calls = 0

    def run(self, traj, fix_map, seed):
        self.calls += 1
        fix = fix_map.get("s1")
        flipped = bool(fix and fix.new_content == "correct")
        return ReplayResult(seed=seed, final_answer="correct" if flipped else "wrong",
                            flipped=flipped)


def _traj() -> Trajectory:
    return Trajectory(
        trajectory_id="stub",
        gaia_task_id="stub",
        query="",
        oracle_answer="correct",
        agent_final_answer="wrong",
        agent_model="test",
        nodes=[Node(step_id="s1", level="worker", role="A",
                    action_type="reasoning", action_content="x")],
    )


def test_empty_fix_map_no_flip(tmp_path):
    cache = ReplayCache(db_path=tmp_path / "c.sqlite")
    replay = BoundedReplay(backend=_StubBackend(), cache=cache)
    results = replay.run(_traj(), {}, [11, 17])
    assert all(not r.flipped for r in results)


def test_correct_fix_flips(tmp_path):
    cache = ReplayCache(db_path=tmp_path / "c.sqlite")
    replay = BoundedReplay(backend=_StubBackend(), cache=cache)
    fix = Fix(node_id="s1", predicted_type="REASONING",
              modified_field="reasoning", new_content="correct", fixer_rationale="")
    results = replay.run(_traj(), {"s1": fix}, [11, 17, 23])
    assert all(r.flipped for r in results)
    assert BoundedReplay.flip_rate(results) == 1.0


def test_cache_avoids_recomputation(tmp_path):
    backend = _StubBackend()
    cache = ReplayCache(db_path=tmp_path / "c.sqlite")
    replay = BoundedReplay(backend=backend, cache=cache)
    fix = Fix(node_id="s1", predicted_type="REASONING",
              modified_field="reasoning", new_content="correct", fixer_rationale="")
    replay.run(_traj(), {"s1": fix}, [11])
    replay.run(_traj(), {"s1": fix}, [11])
    assert backend.calls == 1, "second call with same key must hit cache"
