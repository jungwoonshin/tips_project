"""Stage 2b: oracle-blind fix proposals, one Sonnet call per candidate."""

from __future__ import annotations

import logging
import re

from tips_v3 import checkpoint
from tips_v3.config import FIX_SAMPLES_PER_NODE, FIX_TEMPERATURE, TYPE_ALLOWED_FIELDS
from tips_v3.io.schema import Candidate, Fix, Trajectory
from tips_v3.llm.prompts import fix_2b
from tips_v3.llm.sonnet_client import SonnetClient, parse_json

log = logging.getLogger(__name__)

STAGE = "stage2b"


_FIELD_RE = re.compile(r'"modified_field"\s*:\s*"([^"]+)"')
_RATIONALE_RE = re.compile(r'"fixer_rationale"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _extract_new_content(text: str) -> str | None:
    """Recover the value of `new_content` even if the overall JSON is malformed.

    We scan for the key `"new_content":` and take everything up to the next
    unescaped `"` that is followed by a `,` or `}` (the JSON string's closing
    quote). This is robust to unescaped control characters Sonnet sometimes
    embeds when generating code (e.g. Unlambda backticks)."""
    key = '"new_content":'
    idx = text.find(key)
    if idx == -1:
        return None
    # Find the opening quote of the string value.
    q = text.find('"', idx + len(key))
    if q == -1:
        return None
    # Walk forward, tracking escapes, until we find a closing quote that is
    # followed (after whitespace) by ',' or '}'.
    i = q + 1
    out_chars: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == '\\' and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in '"\\/':
                out_chars.append(nxt); i += 2; continue
            if nxt == 'n': out_chars.append('\n'); i += 2; continue
            if nxt == 't': out_chars.append('\t'); i += 2; continue
            if nxt == 'r': out_chars.append('\r'); i += 2; continue
            # Unknown escape — keep literal
            out_chars.append(ch); i += 1; continue
        if ch == '"':
            # Peek ahead: is this the end-of-string delimiter?
            j = i + 1
            while j < len(text) and text[j] in ' \t\r\n':
                j += 1
            if j < len(text) and text[j] in ',}':
                return "".join(out_chars)
            # Otherwise Sonnet embedded a stray unescaped quote. Keep it.
            out_chars.append(ch); i += 1; continue
        out_chars.append(ch); i += 1
    # Ran off the end — return what we got so far if any
    return "".join(out_chars) if out_chars else None


def _parse_one(text: str, candidate: Candidate) -> Fix | None:
    import json as _json
    try:
        obj = parse_json(text)
        pf = obj.get("proposed_fix") or {}
        field = pf.get("modified_field")
        content = pf.get("new_content")
        rationale = obj.get("fixer_rationale") or ""
    except Exception as exc:
        # Fallback: strict JSON parse failed (usually because of unescaped
        # characters in new_content). Try regex-based field extraction.
        log.warning("stage2b strict parse failed for %s: %s — falling back to regex",
                    candidate.node_id, exc)
        field_m = _FIELD_RE.search(text)
        content = _extract_new_content(text)
        rat_m = _RATIONALE_RE.search(text)
        field = field_m.group(1) if field_m else None
        rationale = rat_m.group(1) if rat_m else ""
        if not field or content is None:
            log.warning("stage2b regex fallback also failed for %s", candidate.node_id)
            return None
        log.info("stage2b recovered fix for %s via regex fallback", candidate.node_id)

    if not field or content is None:
        return None
    if isinstance(content, (dict, list)):
        content = _json.dumps(content, ensure_ascii=False)
    else:
        content = str(content)
    return Fix(
        node_id=candidate.node_id,
        predicted_type=candidate.predicted_type,
        modified_field=field,
        new_content=content,
        fixer_rationale=rationale,
    )


def _best_sample(samples: list[Fix], candidate: Candidate) -> Fix | None:
    """Pick the first sample whose (field, content) satisfies type and
    token-diff constraints — the ones Stage 3 would accept.

    Falls back to the first parsed sample if none cleanly pass, so Stage 3
    can still record the rejection reason."""
    if not samples:
        return None
    allowed = TYPE_ALLOWED_FIELDS.get(candidate.predicted_type, set())
    for s in samples:
        if s.modified_field in allowed:
            return s
    return samples[0]


def propose(traj: Trajectory, candidates: list[Candidate], client: SonnetClient) -> list[Fix]:
    cached = checkpoint.load(traj.trajectory_id, STAGE)
    if cached is not None:
        return [Fix(**f) for f in cached]

    # Build a flat batch: FIX_SAMPLES_PER_NODE messages per candidate.
    flat_messages = []
    flat_candidates: list[Candidate] = []
    for cand in candidates:
        msg = fix_2b.build(traj, cand)
        fix_2b.assert_oracle_absent(msg, traj.oracle_answer)
        for _ in range(FIX_SAMPLES_PER_NODE):
            flat_messages.append(msg)
            flat_candidates.append(cand)

    texts = client.batch(flat_messages, temperature=FIX_TEMPERATURE, max_tokens=1024)

    # Group samples by candidate, pick best per candidate.
    per_node_samples: dict[str, list[Fix]] = {}
    for cand, text in zip(flat_candidates, texts):
        fix = _parse_one(text, cand)
        if fix is not None:
            per_node_samples.setdefault(cand.node_id, []).append(fix)

    fixes: list[Fix] = []
    for cand in candidates:
        samples = per_node_samples.get(cand.node_id, [])
        chosen = _best_sample(samples, cand)
        if chosen is not None:
            fixes.append(chosen)
        log.info("stage2b %s/node %s: %d samples → %s",
                 traj.trajectory_id, cand.node_id, len(samples),
                 "kept" if chosen else "none")

    checkpoint.save(traj.trajectory_id, STAGE, [f.__dict__ for f in fixes])
    log.info("stage2b %s: %d fixes", traj.trajectory_id, len(fixes))
    return fixes
