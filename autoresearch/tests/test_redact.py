"""Oracle-blindness: redact.py must strip ground_truth and
agent_original_final_answer from every dev-set artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch import DEV_TRAJECTORIES, REPO_ROOT
from autoresearch.redact import assert_no_oracle_leak, build_dev_snapshot, redact_answer


ANSWERS_DIR = REPO_ROOT / "output" / "owl_counterfactual_v4" / "answers"


def _iter_dev_originals():
    for tid in DEV_TRAJECTORIES:
        p = ANSWERS_DIR / f"{tid}.json"
        if p.exists():
            yield json.loads(p.read_text())


def test_redact_removes_oracle_fields():
    for orig in _iter_dev_originals():
        red = redact_answer(orig)
        assert red["ground_truth"] == "<REDACTED>"
        assert red["agent_original_final_answer"] == "<REDACTED>"


def test_redact_no_oracle_substring_leak():
    for orig in _iter_dev_originals():
        red = redact_answer(orig)
        assert_no_oracle_leak(red, orig)


def test_build_dev_snapshot_is_oracle_blind():
    snap = build_dev_snapshot(DEV_TRAJECTORIES)
    # Every published dev trajectory should appear; missing is OK for smoke.
    assert isinstance(snap, list)
    for orig, red in zip(_iter_dev_originals(), snap):
        assert red["ground_truth"] == "<REDACTED>"
        assert_no_oracle_leak(red, orig)
