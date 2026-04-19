"""Stage 2a detection prompt (oracle-visible)."""

from __future__ import annotations

import json

from tips_v3.io.schema import Trajectory
from tips_v3.llm.sonnet_client import Message


SYSTEM = """You are an error-attribution analyst for trajectories produced by an LLM agent \
solving GAIA benchmark tasks. You are given the full trajectory and the ground-truth \
oracle answer. The trajectory's final answer is wrong.

Your job: identify the agent nodes whose decisions, if corrected, would plausibly \
propagate to a correct final answer. For each node you identify, classify the error \
using the taxonomy below, give a rationale, and emit a diagnostic_hint describing the \
MECHANISM of the error.

STRICT CONSTRAINT ON diagnostic_hint:
  - Describe what the agent did wrong mechanistically (e.g., "query was too narrow",
    "tool call missing parameter", "conclusion contradicted by retrieved evidence").
  - DO NOT include or paraphrase the oracle answer.
  - DO NOT state what the correct answer is or what the correct value should be.
  - The hint will be passed to a separate model that does NOT see the oracle; leaking
    the answer invalidates the downstream pipeline.

Taxonomy (pick exactly one per node):
  REASONING              inference step draws wrong conclusion from correct evidence
  SEARCH                 search query fails to surface needed information
  TOOL                   wrong tool chosen, or tool args malformed / off-target
  INFORMATION            cited/used source is wrong or misinterpreted
  PLANNING               planner decomposes task poorly or delegates wrong subgoal
  PREMATURE_TERMINATION  agent stops before gathering sufficient evidence

Output JSON in this exact shape (top-level object, candidates ordered by
severity descending):
  {"candidates": [
     {"node_id": "...", "level": "...", "role": "...",
      "predicted_type": "...", "rationale": "...", "diagnostic_hint": "..."}
  ]}

Emit JSON only, no prose."""


def _trajectory_payload(traj: Trajectory) -> list[dict]:
    return [
        {
            "step_id": n.step_id,
            "level": n.level,
            "role": n.role,
            "action_type": n.action_type,
            "action_content": n.action_content[:2000],
            "tool_name": n.tool_name,
            "tool_args": n.tool_args,
            "observation": (n.observation or "")[:1000] if n.observation else None,
        }
        for n in traj.nodes
    ]


def build(traj: Trajectory) -> Message:
    user = (
        f"TRAJECTORY:\n\n{json.dumps(_trajectory_payload(traj), indent=2, ensure_ascii=False)}\n\n"
        f"ORACLE ANSWER: {traj.oracle_answer}\n\n"
        f"AGENT FINAL ANSWER (incorrect): {traj.agent_final_answer}\n\n"
        "Emit the JSON array now."
    )
    return Message(system=SYSTEM, user=user)
