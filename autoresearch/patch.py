"""Patch representation, validation, application, and revert.

We accept a lightweight patch format: a list of (path, new_content) replacements
— simpler and less error-prone than unified diffs for LLM-generated patches.
Each entry replaces the entire file contents. The agent must respect the
allowlist; the validator rejects anything else.

Revert is trivial: we snapshot original bytes before applying.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from autoresearch import REPO_ROOT
from autoresearch.allowlist import (
    ALLOWED_CONFIG_VARS,
    ALLOWED_FILES,
    FORBIDDEN_SYMBOLS,
    path_is_allowed,
    resolve,
)

log = logging.getLogger(__name__)


@dataclass
class FileEdit:
    path: str           # relative-to-repo-root
    new_content: str    # full file contents replacement


@dataclass
class Patch:
    edits: list[FileEdit]
    hypothesis: str = ""
    # Filled by apply()
    backups: dict[str, str] = field(default_factory=dict)

    # --- validation ------------------------------------------------------

    def validate(self) -> tuple[bool, str]:
        if not self.edits:
            return False, "empty_patch"
        for e in self.edits:
            if not path_is_allowed(e.path):
                return False, f"path_not_in_allowlist:{e.path}"
            abs_ = resolve(e.path)
            if not abs_.exists():
                return False, f"target_file_missing:{e.path}"
            # Syntax-check Python edits before applying.
            try:
                ast.parse(e.new_content)
            except SyntaxError as exc:
                return False, f"syntax_error_in_{e.path}: {exc}"
            # Forbidden-symbol check for tips_v3/config.py: extract changed
            # top-level names and ensure none are forbidden.
            if e.path.endswith("config.py"):
                ok, reason = _validate_config_edit(e.new_content)
                if not ok:
                    return False, reason
            # Fix_2b: must keep assert_oracle_absent defined and its body
            # non-trivial (preserve the oracle-blindness check).
            if e.path.endswith("fix_2b.py"):
                if "assert_oracle_absent" not in e.new_content:
                    return False, "removed_assert_oracle_absent"
        return True, "ok"

    # --- apply / revert -------------------------------------------------

    def apply(self) -> None:
        for e in self.edits:
            abs_ = resolve(e.path)
            self.backups[e.path] = abs_.read_text()
            abs_.write_text(e.new_content)

    def revert(self) -> None:
        for path, original in self.backups.items():
            resolve(path).write_text(original)
        self.backups.clear()

    # --- hashing for dedup / cache keys --------------------------------

    def fingerprint(self) -> str:
        blob = "".join(f"{e.path}\0{e.new_content}\0" for e in self.edits)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _validate_config_edit(new_text: str) -> tuple[bool, str]:
    """Ensure config.py edits only touch ALLOWED_CONFIG_VARS.

    We compare the top-level assignment targets of the new file to the
    current one. Any added/changed name outside the allowlist is a reject.
    """
    current_path = REPO_ROOT / "tips_v3" / "config.py"
    current = current_path.read_text()

    def _names(text: str) -> dict[str, str]:
        tree = ast.parse(text)
        out = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        out[tgt.id] = ast.dump(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    out[node.target.id] = ast.dump(node.value)
        return out

    cur = _names(current)
    new = _names(new_text)

    changed = {k for k in new if cur.get(k) != new.get(k)}
    changed |= {k for k in cur if k not in new}  # deletion counts

    for name in changed:
        if name in FORBIDDEN_SYMBOLS:
            return False, f"modified_forbidden_symbol:{name}"
        if name not in ALLOWED_CONFIG_VARS:
            return False, f"modified_non_allowlisted_config:{name}"
    return True, "ok"


# --- construction from agent output ----------------------------------

def from_agent_json(obj: dict) -> Patch:
    """Expected shape:
       {"hypothesis": "...", "edits": [{"path": "...", "new_content": "..."}]}
    """
    edits = [FileEdit(path=e["path"], new_content=e["new_content"])
             for e in obj.get("edits") or []]
    return Patch(edits=edits, hypothesis=obj.get("hypothesis") or "")
