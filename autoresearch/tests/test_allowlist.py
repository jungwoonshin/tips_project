"""Allowlist: patch validator must reject forbidden paths and symbols."""

from __future__ import annotations

import textwrap

import pytest

from autoresearch.allowlist import resolve
from autoresearch.patch import FileEdit, Patch


def test_rejects_path_outside_allowlist():
    p = Patch(edits=[FileEdit(path="tips_v3/replay/bounded_replay.py",
                              new_content="x = 1\n")])
    ok, reason = p.validate()
    assert not ok
    assert "allowlist" in reason


def test_rejects_seeds_modification_in_config():
    cur = resolve("tips_v3/config.py").read_text()
    # Surgically replace SEEDS_GREEDY_INITIAL if present; otherwise synthesize
    # a trivial-but-illegal edit.
    bad = cur
    if "SEEDS_GREEDY_INITIAL" in bad:
        bad = bad.replace("SEEDS_GREEDY_INITIAL", "SEEDS_GREEDY_INITIAL_BAD")
    else:
        bad = bad + "\nSEEDS_GREEDY_INITIAL = [1, 2, 3]\n"
    p = Patch(edits=[FileEdit(path="tips_v3/config.py", new_content=bad)])
    ok, reason = p.validate()
    assert not ok


def test_accepts_allowlisted_temperature_change():
    cur = resolve("tips_v3/config.py").read_text()
    bad = cur.replace("DETECTION_TEMPERATURE = 0.7", "DETECTION_TEMPERATURE = 0.5")
    if bad == cur:
        pytest.skip("DETECTION_TEMPERATURE not found at expected value")
    p = Patch(edits=[FileEdit(path="tips_v3/config.py", new_content=bad)])
    ok, reason = p.validate()
    assert ok, f"expected accept, got reject: {reason}"


def test_rejects_removal_of_assert_oracle_absent():
    cur = resolve("tips_v3/llm/prompts/fix_2b.py").read_text()
    bad = cur.replace("def assert_oracle_absent", "def _removed_assert_oracle")
    p = Patch(edits=[FileEdit(path="tips_v3/llm/prompts/fix_2b.py", new_content=bad)])
    ok, reason = p.validate()
    assert not ok
    assert "assert_oracle_absent" in reason


def test_rejects_syntax_error():
    bad = "def broken(:\n    pass\n"
    p = Patch(edits=[FileEdit(path="tips_v3/llm/prompts/fix_2b.py", new_content=bad)])
    ok, reason = p.validate()
    assert not ok
    assert "syntax" in reason
