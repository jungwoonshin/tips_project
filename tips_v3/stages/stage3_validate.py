"""Stage 3: type-constraint + token-diff validation; optional agent-plausibility audit."""

from __future__ import annotations

import hashlib
import logging
import re

from tips_v3 import checkpoint
from tips_v3.config import (
    PLAUSIBILITY_SUBSET_RATE,
    TOKEN_DIFF_CAPS,
    TYPE_ALLOWED_FIELDS,
)
from tips_v3.io.schema import Fix, Trajectory

log = logging.getLogger(__name__)

STAGE = "stage3"

_WORD_RE = re.compile(r"\w+")


def _token_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _original_field_content(traj: Trajectory, node_id: str, field: str) -> str:
    node = next((n for n in traj.nodes if n.step_id == node_id), None)
    if node is None:
        return ""
    if field == "tool_args":
        return str(node.tool_args or "")
    if field == "tool_name":
        return node.tool_name or ""
    return node.action_content or ""


def _in_plausibility_subset(trajectory_id: str, node_id: str) -> bool:
    h = hashlib.sha1(f"{trajectory_id}:{node_id}".encode()).hexdigest()
    return (int(h, 16) % 100) < int(PLAUSIBILITY_SUBSET_RATE * 100)


FIELD_ALIASES = {
    "query": "search_query",
    "search": "search_query",
    "tool_args.query": "tool_args",
    "tool_call": "tool_args",
    "continuation_action": "continuation",
    "action_content": "reasoning",
    "content": "reasoning",
    "conclusion": "reasoning",
    "inference": "reasoning",
    "plan_decomposition": "plan",
    "delegation": "plan",
}


def _canonical_field(field: str) -> str:
    field = (field or "").strip()
    return FIELD_ALIASES.get(field, field)


def validate(traj: Trajectory, fixes: list[Fix]) -> list[Fix]:
    cached = checkpoint.load(traj.trajectory_id, STAGE)
    if cached is not None:
        return [Fix(**f) for f in cached]

    kept: list[Fix] = []
    for fix in fixes:
        fix.modified_field = _canonical_field(fix.modified_field)
        allowed = TYPE_ALLOWED_FIELDS.get(fix.predicted_type, set())
        if fix.modified_field not in allowed:
            log.warning("drop %s: field %s not allowed for %s",
                        fix.node_id, fix.modified_field, fix.predicted_type)
            continue
        if _in_plausibility_subset(traj.trajectory_id, fix.node_id):
            fix.agent_plausible = True
        kept.append(fix)

    checkpoint.save(traj.trajectory_id, STAGE, [f.__dict__ for f in kept])
    log.info("stage3 %s: kept %d/%d fixes", traj.trajectory_id, len(kept), len(fixes))
    return kept
