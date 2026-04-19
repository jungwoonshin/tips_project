"""Stage 2b fix-proposal prompt (oracle-blind, per-node)."""

from __future__ import annotations

import json

from tips_v3.io.schema import Candidate, Trajectory
from tips_v3.llm.sonnet_client import Message


TYPE_CONSTRAINTS = """TYPE CONSTRAINTS (based on predicted_type):
  REASONING              modify ONLY the single decisive inference at this step —
                         the one sentence or clause whose change redirects the
                         trajectory. Do NOT rewrite the whole derivation. If
                         this is the TERMINAL reasoning step and no further
                         tool call or follow-up is needed, you MAY end with
                         'FINAL ANSWER: <value>' — the replay engine honors
                         terminal fixes directly when they conclude.
  SEARCH                 modify the search call: rewrite the query string AND/OR
                         swap the search backend among {search_tavily,
                         search_duckduckgo, search_wiki}. search_wiki is often
                         the right choice when the answer lives on a specific
                         Wikipedia entity. Do not change surrounding reasoning.
  TOOL                   modify tool name and/or tool arguments.
                         Do not change the reasoning text before or after.
  INFORMATION            modify the cited source or how retrieved content is
                         referenced/interpreted in reasoning. Do NOT fabricate
                         tool-result content — if the retrieval itself is the
                         problem, classify the error as SEARCH or TOOL and
                         change the tool call instead, so the replay can run
                         the tool live and observe a real result.
  PLANNING               modify subgoal decomposition or task delegation (planner only).
                         Worker actions are untouched.
  PREMATURE_TERMINATION  append ONE continuation action consistent with prior history.
                         Do not rewrite history."""


SYSTEM = f"""You are proposing a single-node repair for an LLM agent trajectory. You see \
ONLY the trajectory up to and including the target node. You do NOT see the oracle \
answer, and you do NOT see what happened after this node.

Behave as if you were the agent itself, at this moment, about to emit the target \
node's content, with one hint from a reviewer about what went wrong.

CAPABILITY CAP:
  Stay within the action surface the original agent could have produced: same tool
  schemas, same output format, lengths similar to the agent's empirical distribution
  at this step type. Do not invent new tools, fields, or phrasings.

{TYPE_CONSTRAINTS}

NO TOKEN-DIFF CAP: your fix may be as long as needed to redirect the trajectory,
but the capability cap above still applies — stay within the action surface the
original agent could produce.

Output exactly this JSON (wrapped in ```json ... ``` fences, no prose):
{{
  "node_id": "...",
  "predicted_type": "...",
  "proposed_fix": {{ "modified_field": "...", "new_content": "..." }},
  "fixer_rationale": "short justification using only pre-error context"
}}"""


def _pre_error_payload(traj: Trajectory, node_id: str) -> list[dict]:
    out: list[dict] = []
    for n in traj.nodes:
        out.append(
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
        )
        if n.step_id == node_id:
            break
    return out


_EMPTY_MARKERS = ("'results': []", '"results": []', "no results", "not found")


def _empty_observation_hint(traj: Trajectory, node_id: str) -> str | None:
    """Detect whether a given tool returned empty/useless results in the
    pre-error trajectory. If so, tell Sonnet to prefer a backend swap."""
    target = next((n for n in traj.nodes if n.step_id == node_id), None)
    if target is None or target.action_type != "tool_call":
        return None
    tool = target.tool_name
    if not tool:
        return None
    empties = 0
    for n in traj.nodes:
        if n.step_id == node_id:
            break
        if n.action_type == "tool_result" and n.observation:
            obs = n.observation.lower()
            if tool and tool.lower() in obs and any(m in obs for m in _EMPTY_MARKERS):
                empties += 1
    if empties == 0:
        return None
    alternatives = {
        "search_tavily": "search_duckduckgo, search_wiki",
        "search_duckduckgo": "search_tavily, search_wiki",
        "search_wiki": "search_tavily, search_duckduckgo",
    }.get(tool, "a different search backend")
    return (
        f"NOTE: `{tool}` has already returned empty or irrelevant results "
        f"{empties} time(s) in the pre-error trajectory. Strongly prefer "
        f"swapping the backend to {alternatives} over re-wording the same "
        f"tool's query — a different index is far more likely to surface the "
        f"needed evidence."
    )


def build(traj: Trajectory, candidate: Candidate) -> Message:
    """Build the Stage 2b request for one candidate node. Oracle is absent by construction."""
    pre = _pre_error_payload(traj, candidate.node_id)
    empty_hint = _empty_observation_hint(traj, candidate.node_id)
    hint_block = f"\n{empty_hint}\n" if empty_hint else ""
    user = (
        f"TRAJECTORY UP TO AND INCLUDING THE TARGET NODE:\n\n"
        f"{json.dumps(pre, indent=2, ensure_ascii=False)}\n\n"
        f"TARGET NODE:\n"
        f"  node_id: {candidate.node_id}\n"
        f"  level: {candidate.level}\n"
        f"  role: {candidate.role}\n"
        f"  predicted_type: {candidate.predicted_type}\n"
        f"{hint_block}\n"
        f"DIAGNOSTIC HINT (reviewer note, oracle-blind): {candidate.diagnostic_hint}\n\n"
        "Emit the JSON object now."
    )
    return Message(system=SYSTEM, user=user)


def assert_oracle_absent(msg: Message, oracle_answer: str) -> None:
    """Oracle must not appear in the system prompt or the diagnostic_hint line.

    The trajectory body (embedded in the user turn) is the agent's own view and
    may legitimately contain the oracle answer — e.g. the agent saw the correct
    value in a tool result but then chose wrong. That is NOT leakage into the
    fixer: the fixer is only entitled to whatever the agent saw pre-error.

    Skips very short oracles (<3 chars) to avoid spurious matches on
    numbers/words that appear naturally in text (e.g. '41', 'yes'); matching
    policy is consistent with Stage 2a's hint filter."""
    import re
    oracle = oracle_answer.strip().lower()
    if not oracle or len(oracle) < 3:
        return
    pat = r"\b" + re.escape(oracle) + r"\b"
    if re.search(pat, msg.system.lower()):
        raise AssertionError("oracle answer leaked into Stage 2b system prompt")
    for line in msg.user.splitlines():
        if line.startswith("DIAGNOSTIC HINT") and re.search(pat, line.lower()):
            raise AssertionError("oracle answer leaked into Stage 2b diagnostic hint")
