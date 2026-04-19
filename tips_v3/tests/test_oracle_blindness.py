"""BLOCKING TEST: Stage 2b request must never contain oracle answer.

Any change to the Stage 2b request builder that lands with this test failing
invalidates the oracle-blindness guarantee.
"""

from __future__ import annotations

import pytest

from tips_v3.io.schema import Candidate, Node, Trajectory
from tips_v3.llm.prompts import detect_2a, fix_2b


def _make_traj(oracle: str = "SECRET_ORACLE_ANSWER_42") -> Trajectory:
    return Trajectory(
        trajectory_id="t1",
        gaia_task_id="t1",
        query="what is the secret?",
        oracle_answer=oracle,
        agent_final_answer="WRONG",
        agent_model="test",
        nodes=[
            Node(step_id="1", level="worker", role="Assistant",
                 action_type="reasoning", action_content="I will search."),
            Node(step_id="2", level="worker", role="Assistant",
                 action_type="tool_call", action_content="search(foo)",
                 tool_name="search", tool_args={"q": "foo"}),
            Node(step_id="3", level="worker", role="Assistant",
                 action_type="final_answer", action_content="FINAL ANSWER: WRONG"),
        ],
    )


def _candidate() -> Candidate:
    return Candidate(
        node_id="2",
        level="worker",
        role="Assistant",
        predicted_type="SEARCH",
        confidence=1.0,
        rationale="query too narrow",
        diagnostic_hint="the query was too narrow to surface the needed information",
    )


def test_stage2a_contains_oracle():
    traj = _make_traj()
    msg = detect_2a.build(traj)
    assert traj.oracle_answer in msg.user, "Stage 2a MUST show oracle (detection is oracle-visible)"


def test_stage2b_excludes_oracle():
    traj = _make_traj()
    msg = fix_2b.build(traj, _candidate())
    assert traj.oracle_answer not in msg.system
    assert traj.oracle_answer not in msg.user
    fix_2b.assert_oracle_absent(msg, traj.oracle_answer)


def test_stage2b_excludes_post_error_steps():
    traj = _make_traj()
    msg = fix_2b.build(traj, _candidate())
    assert "FINAL ANSWER: WRONG" not in msg.user, "Stage 2b must not see post-error trajectory"


def test_assert_oracle_absent_raises_on_system_leak():
    traj = _make_traj()
    leaky = fix_2b.Message(system="the answer is SECRET_ORACLE_ANSWER_42", user="")
    with pytest.raises(AssertionError):
        fix_2b.assert_oracle_absent(leaky, traj.oracle_answer)


def test_assert_oracle_absent_raises_on_hint_leak():
    traj = _make_traj()
    leaky = fix_2b.Message(
        system="",
        user="TARGET NODE: x\n\nDIAGNOSTIC HINT: the answer is SECRET_ORACLE_ANSWER_42\n",
    )
    with pytest.raises(AssertionError):
        fix_2b.assert_oracle_absent(leaky, traj.oracle_answer)


def test_assert_oracle_absent_allows_trajectory_embedded():
    """Trajectory body may legitimately contain oracle (agent saw it in a
    tool result); that is not leakage into the fixer."""
    traj = _make_traj()
    ok = fix_2b.Message(
        system="",
        user='TRAJECTORY: [{"observation": "SECRET_ORACLE_ANSWER_42"}]\n\n'
             "DIAGNOSTIC HINT: the query was too narrow\n",
    )
    fix_2b.assert_oracle_absent(ok, traj.oracle_answer)
