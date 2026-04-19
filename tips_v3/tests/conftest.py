"""Isolate all tests from the production checkpoint/output directories."""

from __future__ import annotations

import pytest

from tips_v3 import config


@pytest.fixture(autouse=True)
def _isolate_output_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "CKPT_DIR", tmp_path / "out" / "_ckpt")
    monkeypatch.setattr(config, "DIFFICULT_DIR", tmp_path / "out" / "_difficult")
    monkeypatch.setattr(config, "ANSWERS_DIR", tmp_path / "out" / "answers")
    monkeypatch.setattr(config, "REPLAY_CACHE_DB", tmp_path / "out" / "_replay.sqlite")
    config.ensure_output_dirs()
    yield
