"""The meta-research agent — proposes patches via Sonnet 4.5.

Oracle-blindness: the agent prompt NEVER includes oracle_answer or
agent_original_final_answer strings. It sees:
  - current pipeline source (allowlist files, full contents)
  - recent iteration history (hypothesis + score + kept/reverted)
  - redacted dev-set artifacts (replay outputs, flip booleans, fix contents)
  - a paradigm-shift nudge every N iterations
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from autoresearch import REPO_ROOT
from autoresearch.allowlist import ALLOWED_CONFIG_VARS, ALLOWED_FILES, resolve
from autoresearch.patch import Patch, from_agent_json
from tips_v3.llm.sonnet_client import Message, SonnetClient, parse_json

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a research agent iterating on an LLM-agent error-repair \
pipeline called TIPS. TIPS attempts to flip the answer of failed GAIA trajectories \
(produced by OWL + Gemma 4 31B) by detecting error steps with Sonnet 4.5, proposing \
oracle-blind fixes, and replaying the agent with fixes spliced in. Your job is to \
propose ONE change to the pipeline's prompts or hyperparameters that will raise the \
flip rate on a held-out dev set.

ORACLE-BLINDNESS INVARIANT (hard constraint):
  - You will NEVER be shown the ground-truth oracle answer for any trajectory.
  - The replay harness and scoring function are frozen — you cannot edit them.
  - Your edits may ONLY touch the files in the allowlist below; anything else is
    rejected by the patch validator.

OUTPUT (wrap in ```json fences):
{
  "hypothesis": "<one sentence explaining the intended mechanism>",
  "edits": [
    {"path": "<allowlisted path>", "new_content": "<FULL file contents>"}
  ]
}

IMPORTANT:
  - Each edit REPLACES the entire file. Include all unchanged code verbatim —
    the patch applier does not merge diffs.
  - Do not modify files outside the allowlist.
  - Do not remove the `assert_oracle_absent` function in fix_2b.py.
  - Keep Python syntax valid; the validator parses every edit with ast.parse.
  - When editing config.py, only change values of ALLOWED_CONFIG_VARS listed.

Propose ONE change. Small, mechanistically-explained changes tend to work better
than sweeping rewrites. But every 20 iterations the orchestrator will ask you for
a "paradigm shift" — a deliberately larger structural change."""


def _read_allowed_files() -> dict[str, str]:
    out = {}
    for rel in sorted(ALLOWED_FILES):
        p = resolve(rel)
        if p.exists():
            out[rel] = p.read_text()
    return out


def _format_history(recent: list[dict]) -> str:
    lines = []
    for entry in recent:
        parts = [f"iter={entry.get('iter')}"]
        if "score" in entry:
            parts.append(f"score={entry.get('score')}")
        if "kept" in entry:
            parts.append(f"kept={entry.get('kept')}")
        if entry.get("hypothesis"):
            parts.append(f"hypothesis={entry['hypothesis'][:180]!r}")
        if entry.get("reason"):
            parts.append(f"reason={entry['reason']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines) or "(no prior iterations)"


def build_prompt(
    *,
    recent_history: list[dict],
    dev_snapshot: list[dict],
    paradigm_shift: bool,
    baseline_score: float,
    current_score: float | None,
) -> Message:
    allowed_files = _read_allowed_files()
    allowed_files_dump = "\n\n".join(
        f"### {path}\n```python\n{body}\n```"
        for path, body in allowed_files.items()
    )
    hist_dump = _format_history(recent_history)
    dev_dump = json.dumps(dev_snapshot, ensure_ascii=False, indent=2)

    user = f"""CURRENT DEV FLIP RATE: {current_score if current_score is not None else '(not yet measured)'}
BASELINE DEV FLIP RATE: {baseline_score}
ALLOWED CONFIG VARS (for edits to tips_v3/config.py):
  {sorted(ALLOWED_CONFIG_VARS)}

--- ALLOWED FILES (current contents) ---

{allowed_files_dump}

--- RECENT ITERATION HISTORY ---
{hist_dump}

--- DEV-SET SNAPSHOT (oracle-redacted) ---
This is what the current pipeline is producing on the dev set. The ground truth
and agent-original-answer fields have been stripped. You can see per-seed replay
outputs and whether each seed flipped, but NOT what the correct answer is.

{dev_dump}

--- TASK ---
{'PARADIGM-SHIFT ITERATION: propose a deliberately LARGER, structurally different change than usual. Consider swapping TYPE_PRIORITY entirely, rewriting a whole prompt, or dropping a validation filter. Explain the bet.' if paradigm_shift else 'Propose one targeted change that mechanistically addresses a specific failure you observe in the dev snapshot.'}

Emit the JSON patch now."""
    return Message(system=SYSTEM_PROMPT, user=user)


class ResearchAgent:
    def __init__(self, client: SonnetClient | None = None):
        self.client = client or SonnetClient()

    def propose(
        self,
        *,
        recent_history: list[dict],
        dev_snapshot: list[dict],
        paradigm_shift: bool = False,
        baseline_score: float = 0.0,
        current_score: float | None = None,
        temperature: float = 0.6,
        max_tokens: int = 16384,
    ) -> tuple[str, Patch | None, str]:
        """Returns (hypothesis, patch_or_none, raw_response)."""
        msg = build_prompt(
            recent_history=recent_history,
            dev_snapshot=dev_snapshot,
            paradigm_shift=paradigm_shift,
            baseline_score=baseline_score,
            current_score=current_score,
        )
        raw = self.client.call(msg, temperature=temperature, max_tokens=max_tokens)
        try:
            obj = parse_json(raw)
        except Exception as exc:
            log.warning("agent response parse failure: %s", exc)
            return "", None, raw
        patch = from_agent_json(obj)
        return patch.hypothesis, patch, raw
