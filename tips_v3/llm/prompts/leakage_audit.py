"""Stage 7 F2 leakage audit prompt (batched per trajectory)."""

from __future__ import annotations

import json

from tips_v3.io.schema import Fix
from tips_v3.llm.sonnet_client import Message


SYSTEM = """You are a leakage auditor. Given an oracle answer and a set of fix contents, \
flag any fix whose content restates, paraphrases, or strongly implies the oracle \
answer. A fix that merely points the agent in a useful direction is NOT leakage; a \
fix that contains the answer value itself IS leakage.

Output JSON: {"leaky_fix_ids": [...], "notes": {"<fix_id>": "reason"}}

Wrap in ```json ... ``` fences."""


def build(oracle_answer: str, fixes: list[Fix]) -> Message:
    payload = [
        {"fix_id": f.node_id, "type": f.predicted_type, "content": f.new_content[:1500]}
        for f in fixes
    ]
    user = (
        f"ORACLE ANSWER: {oracle_answer}\n\n"
        f"FIXES:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        "Audit now."
    )
    return Message(system=SYSTEM, user=user)
