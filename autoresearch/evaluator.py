"""Evaluator — runs the TIPS pipeline on a trajectory subset, returns flip_rate.

Reuses tips_v3.run.process_one. Invalidates stage4+ checkpoints + replay cache
before each evaluation so the run actually exercises current prompts/config.
Stage2a/2b checkpoints are kept only when the relevant prompt hasn't changed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from autoresearch import AUTORESEARCH_OUTPUT_DIR, FROZEN_BASELINE_DIR, REPO_ROOT

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ cache

def _prompt_hash() -> str:
    """Hash over all prompt files + relevant config symbols. Used to decide
    whether cached stage2a/stage2b checkpoints are still valid."""
    parts: list[bytes] = []
    for rel in ("tips_v3/llm/prompts/detect_2a.py",
                "tips_v3/llm/prompts/fix_2b.py",
                "tips_v3/llm/prompts/expand.py",
                "tips_v3/llm/prompts/null_fix.py",
                "tips_v3/config.py"):
        p = REPO_ROOT / rel
        if p.exists():
            parts.append(p.read_bytes())
    return hashlib.sha1(b"".join(parts)).hexdigest()[:12]


def _reset_pipeline_state(output_dir: Path, trajectory_ids: Iterable[str]) -> None:
    """Wipe stage4+ checkpoints + replay cache + per-trajectory answers for the
    subset we're about to evaluate. Stages 2a/2b survive only if prompts
    unchanged — we keep them for speed because Stage 2a Sonnet calls are the
    heaviest cost."""
    ckpt_root = output_dir / "_ckpt"
    for tid in trajectory_ids:
        d = ckpt_root / tid
        for name in ("stage4.json", "stage5.json", "stage6.json", "stage7.json", "stage8.json"):
            f = d / name
            if f.exists():
                f.unlink()
        # Delete downstream outputs for this tid
        (output_dir / "answers" / f"{tid}.json").unlink(missing_ok=True)
        (output_dir / "_difficult" / f"{tid}.json").unlink(missing_ok=True)
        (output_dir / f"{tid}.json").unlink(missing_ok=True)

    # Drop replay cache (small; re-fills cheaply)
    cache_db = output_dir / "_replay_cache.sqlite"
    if cache_db.exists():
        cache_db.unlink()


# --------------------------------------------------------- invalidation

def _invalidate_stage2_if_prompts_changed(output_dir: Path,
                                          trajectory_ids: Iterable[str],
                                          last_prompt_hash: str | None) -> str:
    """If any prompt/config file changed since last_prompt_hash, wipe
    stage2a/2b for the affected subset. Returns the new hash."""
    current = _prompt_hash()
    if last_prompt_hash and current == last_prompt_hash:
        return current
    ckpt_root = output_dir / "_ckpt"
    for tid in trajectory_ids:
        d = ckpt_root / tid
        for name in ("stage2a.json", "stage2b.json", "stage3.json"):
            f = d / name
            if f.exists():
                f.unlink()
    log.info("prompt hash changed (%s → %s); invalidated stage2a/2b for %d trajectories",
             last_prompt_hash, current, len(list(trajectory_ids)))
    return current


# ------------------------------------------------------------------ eval

def evaluate(
    trajectory_ids: list[str],
    *,
    output_dir: Path | None = None,
    last_prompt_hash: str | None = None,
    parallel_workers: int = 5,
    use_fast_seeds: bool = True,
) -> dict:
    """Run the TIPS pipeline on `trajectory_ids` and return flip-rate dict.

    Returns:
        {
          "flip_rate": float,
          "per_tid_flipped": {tid: bool},
          "wall_clock_s": float,
          "prompt_hash": str,
        }

    Oracle is never exposed to the caller — this function returns only flip
    booleans. The oracle is consulted inside BoundedReplay via gaia_scorer.
    """
    if output_dir is None:
        output_dir = AUTORESEARCH_OUTPUT_DIR

    # Re-import to pick up any patched config / prompts written by the agent.
    import tips_v3.config  # noqa: F401
    importlib.reload(tips_v3.config)
    from tips_v3 import config as cfg
    # Prompts are re-read each call via `importlib.reload` on the modules that
    # reference them (stage2a_detect etc. import prompt modules eagerly).
    for mod_name in ("tips_v3.llm.prompts.detect_2a",
                     "tips_v3.llm.prompts.fix_2b",
                     "tips_v3.llm.prompts.expand",
                     "tips_v3.llm.prompts.null_fix",
                     "tips_v3.stages.stage2a_detect",
                     "tips_v3.stages.stage2b_fix",
                     "tips_v3.stages.stage3_validate",
                     "tips_v3.stages.stage4_greedy",
                     "tips_v3.stages.stage5_reduce",
                     "tips_v3.stages.stage6_verify",
                     "tips_v3.stages.stage7_filters",
                     "tips_v3.stages.stage8_expand"):
        if mod_name in __import__("sys").modules:
            importlib.reload(__import__("sys").modules[mod_name])

    # Point config at the evaluator output dir (uses dynamic attr access)
    cfg.OUTPUT_DIR = output_dir
    cfg.CKPT_DIR = output_dir / "_ckpt"
    cfg.DIFFICULT_DIR = output_dir / "_difficult"
    cfg.ANSWERS_DIR = output_dir / "answers"
    cfg.REPLAY_CACHE_DB = output_dir / "_replay_cache.sqlite"
    for d in (output_dir, cfg.CKPT_DIR, cfg.DIFFICULT_DIR, cfg.ANSWERS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    new_hash = _invalidate_stage2_if_prompts_changed(output_dir, trajectory_ids, last_prompt_hash)
    _reset_pipeline_state(output_dir, trajectory_ids)

    from tips_v3.io.ingest import iter_trajectories
    from tips_v3.llm.sonnet_client import SonnetClient
    from tips_v3.replay.bounded_replay import BoundedReplay
    from tips_v3.run import process_one

    wanted = set(trajectory_ids)
    trajs = [t for t in iter_trajectories() if t.trajectory_id in wanted]
    if not trajs:
        raise RuntimeError(f"no trajectories found from: {sorted(wanted)}")

    client = SonnetClient()
    replay = BoundedReplay()
    # Shared summary dict (thread-safe-ish — we only ever increment counts
    # outside the worker region, so keep updates local and merge at the end).
    summary_per_tid: dict[str, str] = {}

    t0 = time.time()

    def _run_one(traj):
        local_summary = {"counts": {"total_input": 0, "published": 0,
                                     "expansion_queue": 0, "difficult": 0,
                                     "filtered_by_f2": 0}}
        try:
            process_one(traj, client, replay, local_summary)
        except Exception as exc:
            log.exception("evaluator %s crashed: %s", traj.trajectory_id, exc)
            return traj.trajectory_id, "exception"
        if local_summary["counts"]["published"] >= 1:
            return traj.trajectory_id, "published"
        return traj.trajectory_id, "difficult"

    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        futs = [pool.submit(_run_one, t) for t in trajs]
        for fut in as_completed(futs):
            tid, outcome = fut.result()
            summary_per_tid[tid] = outcome

    per_tid_flipped = {tid: (summary_per_tid.get(tid) == "published")
                        for tid in trajectory_ids}
    flipped = sum(1 for v in per_tid_flipped.values() if v)
    flip_rate = flipped / max(len(trajectory_ids), 1)

    return {
        "flip_rate": flip_rate,
        "per_tid_flipped": per_tid_flipped,
        "wall_clock_s": time.time() - t0,
        "prompt_hash": new_hash,
    }
