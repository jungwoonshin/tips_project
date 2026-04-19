"""Stage 7 F1 null-fix control prompt."""

from __future__ import annotations

import json

from tips_v3.io.schema import Fix, Trajectory
from tips_v3.llm.prompts.fix_2b import TYPE_CONSTRAINTS, _pre_error_payload
from tips_v3.llm.sonnet_client import Message


SYSTEM = f"""You are producing a CONTROL fix for a validity test. Given a target node and \
its error type, produce a same-type but semantically-DIFFERENT fix that is still \
WRONG. The control fix must:
  - Match the token-diff budget for this type (see Stage 2b constraints).
  - Stay within the agent's action surface (same capability cap as the real fixer).
  - NOT accidentally correct the error — it must be a different-but-equally-plausible
    WRONG alternative.

{TYPE_CONSTRAINTS}

Output the same JSON schema as Stage 2b, wrapped in ```json ... ``` fences."""


def build(traj: Trajectory, fix: Fix) -> Message:
    pre = _pre_error_payload(traj, fix.node_id)
    user = (
        f"TRAJECTORY UP TO TARGET NODE:\n\n"
        f"{json.dumps(pre, indent=2, ensure_ascii=False)}\n\n"
        f"TARGET NODE: {fix.node_id}\n"
        f"ERROR TYPE: {fix.predicted_type}\n"
        f"(The real fix has already been proposed; produce a different WRONG alternative.)\n\n"
        "Emit the JSON object now."
    )
    return Message(system=SYSTEM, user=user)
