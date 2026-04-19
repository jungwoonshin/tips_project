"""Per-trajectory stage checkpoints for resumable runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tips_v3 import config


def _ckpt_path(trajectory_id: str, stage: str) -> Path:
    return config.CKPT_DIR / trajectory_id / f"{stage}.json"


def has(trajectory_id: str, stage: str) -> bool:
    return _ckpt_path(trajectory_id, stage).exists()


def load(trajectory_id: str, stage: str) -> Any | None:
    path = _ckpt_path(trajectory_id, stage)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save(trajectory_id: str, stage: str, payload: Any) -> Path:
    path = _ckpt_path(trajectory_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_default))
    os.replace(tmp, path)
    return path


def _default(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def clear_from(trajectory_id: str, stage: str) -> None:
    """Delete this stage's checkpoint and all downstream ones."""
    order = ["stage2a", "stage2b", "stage3", "stage4", "stage5", "stage6", "stage7", "stage8"]
    if stage not in order:
        return
    idx = order.index(stage)
    for s in order[idx:]:
        p = _ckpt_path(trajectory_id, s)
        if p.exists():
            p.unlink()
