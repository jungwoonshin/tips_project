"""Atomic JSON writer for per-trajectory records."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tips_v3.config import ANSWERS_DIR, DIFFICULT_DIR, OUTPUT_DIR
from tips_v3.io.schema import Record


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def write_record(record: Record) -> Path:
    _validate_record(record)
    out = OUTPUT_DIR / f"{record.trajectory_id}.json"
    _atomic_write(out, record.to_dict())
    return out


def write_difficult(trajectory_id: str, payload: dict) -> Path:
    out = DIFFICULT_DIR / f"{trajectory_id}.json"
    _atomic_write(out, payload)
    return out


def write_answer(trajectory_id: str, payload: dict) -> Path:
    out = ANSWERS_DIR / f"{trajectory_id}.json"
    _atomic_write(out, payload)
    return out


def _validate_record(record: Record) -> None:
    if not record.published_sufficient_set:
        raise ValueError(f"{record.trajectory_id}: empty published_sufficient_set")
    md = record.validity_metadata
    if md.get("k_final_verify") != 5:
        raise ValueError(f"{record.trajectory_id}: k_final_verify must be 5")
    if md.get("replay_mode") != "bounded":
        raise ValueError(f"{record.trajectory_id}: replay_mode must be 'bounded'")
    if md.get("leakage_audit_passed") is not True:
        raise ValueError(f"{record.trajectory_id}: leakage_audit_passed must be True")
    flip = md.get("flip_rate_final")
    if flip is None or flip < 0.6:
        raise ValueError(f"{record.trajectory_id}: flip_rate_final {flip} < 0.6")
    for sid in record.published_sufficient_set:
        if sid not in record.per_node:
            raise ValueError(f"{record.trajectory_id}: missing per_node entry for {sid}")
        if sid not in record.fixes_supplementary:
            raise ValueError(f"{record.trajectory_id}: missing fix for {sid}")
