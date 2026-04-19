"""Stage 3 rejects fixes that touch disallowed fields or blow token-diff caps."""

from __future__ import annotations

from tips_v3.io.schema import Fix, Node, Trajectory
from tips_v3.stages.stage3_validate import validate


def _traj() -> Trajectory:
    return Trajectory(
        trajectory_id="tx",
        gaia_task_id="tx",
        query="q",
        oracle_answer="ans",
        agent_final_answer="wrong",
        agent_model="test",
        nodes=[
            Node(step_id="s1", level="worker", role="A",
                 action_type="tool_call", action_content="search(x)",
                 tool_name="search_tavily", tool_args={"q": "original query"}),
        ],
    )


def test_allowed_field_passes():
    fix = Fix(node_id="s1", predicted_type="SEARCH",
              modified_field="search_query",
              new_content="tighter query", fixer_rationale="...")
    assert validate(_traj(), [fix])


def test_disallowed_field_dropped():
    fix = Fix(node_id="s1", predicted_type="SEARCH",
              modified_field="reasoning",
              new_content="new reasoning", fixer_rationale="...")
    assert validate(_traj(), [fix]) == []


def test_token_cap_removed():
    # Token caps were removed — arbitrarily long fixes are now allowed.
    huge = "word " * 500
    fix = Fix(node_id="s1", predicted_type="SEARCH",
              modified_field="search_query",
              new_content=huge, fixer_rationale="...")
    assert validate(_traj(), [fix])  # kept, not dropped
