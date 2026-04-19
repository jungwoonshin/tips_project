"""Stage 8 adversarial re-detection prompt."""

from __future__ import annotations

import json

from tips_v3.io.schema import Candidate, Trajectory
from tips_v3.llm.prompts.detect_2a import _trajectory_payload
from tips_v3.llm.sonnet_client import Message


SYSTEM = """The prior error-detection pass produced a candidate set that was insufficient \
to flip the answer under bounded replay. Act as an ADVERSARIAL reviewer: what error \
nodes did the first pass MISS? Focus on:
  - Early planner decisions whose effects were masked by later worker actions.
  - Subtle information-use errors that look correct but trace back to a wrong source.
  - Premature terminations disguised as confident final answers.

Emit the same JSON object shape as Stage 2a ({"candidates": [...]}), with \
candidates NOT present in the prior pass. Up to 8 new candidates. Same strict \
rule: diagnostic_hint must be answer-free. Emit JSON only, no prose."""


def build(traj: Trajectory, prior: list[Candidate]) -> Message:
    prior_ids = [c.node_id for c in prior]
    user = (
        f"TRAJECTORY:\n\n{json.dumps(_trajectory_payload(traj), indent=2, ensure_ascii=False)}\n\n"
        f"ORACLE ANSWER: {traj.oracle_answer}\n"
        f"AGENT FINAL ANSWER: {traj.agent_final_answer}\n"
        f"PRIOR CANDIDATE node_ids (exclude these): {prior_ids}\n\n"
        "Emit the JSON array now."
    )
    return Message(system=SYSTEM, user=user)
