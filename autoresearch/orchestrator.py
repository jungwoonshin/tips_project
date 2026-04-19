"""Main AutoResearch loop. Reads dev/test split, runs baseline, iterates with
ratchet + overfit watchdog, writes history to disk."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Iterable

from autoresearch import (
    AUTORESEARCH_OUTPUT_DIR,
    AUTORESEARCH_ROOT,
    DEV_TRAJECTORIES,
    FROZEN_BASELINE_DIR,
    RUNS_DIR,
)
from autoresearch.agent import ResearchAgent
from autoresearch.evaluator import evaluate
from autoresearch.history import History
from autoresearch.redact import build_dev_snapshot

log = logging.getLogger(__name__)


def _compute_test_set(dev: Iterable[str]) -> list[str]:
    """Test = all other trajectories in the frozen v4 answers dir that aren't
    in dev. v4 is the source of truth for which trajectories exist."""
    answers_dir = FROZEN_BASELINE_DIR / "answers"
    if not answers_dir.exists():
        return []
    dev_set = set(dev)
    all_tids = sorted(p.stem for p in answers_dir.glob("gaia_validation_*.json"))
    return [t for t in all_tids if t not in dev_set]


# ------------------------------------------------------------------ v5 snapshot

_V5_LIVE_FILES = ("answers", "_difficult", "_replay_cache.sqlite", "summary.json")


def _snapshot_v5(dest: Path) -> None:
    """Snapshot the live portion of v5 so a reverted iteration can restore it."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in _V5_LIVE_FILES:
        src = AUTORESEARCH_OUTPUT_DIR / name
        tgt = dest / name
        if tgt.exists():
            if tgt.is_dir():
                shutil.rmtree(tgt)
            else:
                tgt.unlink()
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, tgt)
            else:
                shutil.copy2(src, tgt)
    # Top-level published-record files (glob)
    for pub in AUTORESEARCH_OUTPUT_DIR.glob("gaia_validation_*.json"):
        shutil.copy2(pub, dest / pub.name)


def _restore_v5(src: Path) -> None:
    """Restore v5's live portion from a snapshot."""
    if not src.exists():
        return
    for name in _V5_LIVE_FILES:
        live = AUTORESEARCH_OUTPUT_DIR / name
        if live.exists():
            if live.is_dir():
                shutil.rmtree(live)
            else:
                live.unlink()
        s = src / name
        if s.exists():
            if s.is_dir():
                shutil.copytree(s, live)
            else:
                shutil.copy2(s, live)
    # Nuke stale published records then re-copy from snapshot
    for pub in AUTORESEARCH_OUTPUT_DIR.glob("gaia_validation_*.json"):
        pub.unlink()
    for pub in src.glob("gaia_validation_*.json"):
        shutil.copy2(pub, AUTORESEARCH_OUTPUT_DIR / pub.name)


def _bootstrap_v5_from_v4() -> None:
    """First-run init: if v5 has no answers yet, seed it from v4 so the agent
    starts with a real baseline instead of an empty dataset."""
    AUTORESEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (AUTORESEARCH_OUTPUT_DIR / "answers").mkdir(exist_ok=True)
    if any((AUTORESEARCH_OUTPUT_DIR / "answers").glob("*.json")):
        return  # already bootstrapped; preserve existing autoresearch state
    log.info("bootstrapping v5 from v4 (first run)")
    for name in _V5_LIVE_FILES:
        src = FROZEN_BASELINE_DIR / name
        tgt = AUTORESEARCH_OUTPUT_DIR / name
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, tgt)
            else:
                shutil.copy2(src, tgt)
    for pub in FROZEN_BASELINE_DIR.glob("gaia_validation_*.json"):
        shutil.copy2(pub, AUTORESEARCH_OUTPUT_DIR / pub.name)
    # Also seed checkpoints so Sonnet stage2a/2b results don't have to re-run
    ckpt_src = FROZEN_BASELINE_DIR / "_ckpt"
    ckpt_tgt = AUTORESEARCH_OUTPUT_DIR / "_ckpt"
    if ckpt_src.exists() and not ckpt_tgt.exists():
        shutil.copytree(ckpt_src, ckpt_tgt)


def run(
    *,
    budget: int = 100,
    ratchet_slack: float = 0.02,
    watchdog_every: int = 20,
    overfit_threshold: float = 0.15,
    paradigm_shift_every: int = 20,
    dev_only: bool = False,
    run_id: str | None = None,
) -> dict:
    run_id = run_id or f"run_{int(time.time())}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    history = History(run_dir / "history.jsonl")
    snapshot_dir = run_dir / "v5_best_snapshot"

    dev_set = list(DEV_TRAJECTORIES)
    test_set = [] if dev_only else _compute_test_set(dev_set)

    log.info("run_id=%s dev=%s test=%d", run_id, dev_set, len(test_set))

    # Seed v5 from v4 on very first autoresearch run; preserve it otherwise.
    _bootstrap_v5_from_v4()

    agent = ResearchAgent()

    # ---- baseline ----
    t0 = time.time()
    res = evaluate(dev_set)
    # First snapshot = baseline state. Any reverted iteration restores to this.
    _snapshot_v5(snapshot_dir)
    baseline = res["flip_rate"]
    prompt_hash = res["prompt_hash"]
    history.append({
        "iter": 0, "kind": "baseline",
        "score": baseline, "kept": True,
        "per_tid": res["per_tid_flipped"],
        "wall_clock_s": res["wall_clock_s"],
        "prompt_hash": prompt_hash,
    })
    log.info("baseline dev flip rate: %.3f", baseline)

    last_score = baseline

    for i in range(1, budget + 1):
        paradigm_shift = (i % paradigm_shift_every == 0)
        dev_snapshot = build_dev_snapshot(dev_set)

        hypothesis, patch, raw = agent.propose(
            recent_history=history.tail(10),
            dev_snapshot=dev_snapshot,
            paradigm_shift=paradigm_shift,
            baseline_score=baseline,
            current_score=last_score,
        )
        # Persist agent raw output
        (run_dir / f"iter_{i:03d}_agent.txt").write_text(raw[:20000])

        if patch is None:
            history.append({"iter": i, "kind": "proposal", "kept": False,
                            "reason": "parse_failure",
                            "hypothesis": hypothesis})
            continue

        ok, reason = patch.validate()
        if not ok:
            history.append({"iter": i, "kind": "proposal", "kept": False,
                            "reason": f"invalid_patch:{reason}",
                            "hypothesis": hypothesis,
                            "patch_fingerprint": patch.fingerprint(),
                            "edited_paths": [e.path for e in patch.edits]})
            continue

        patch.apply()
        try:
            res = evaluate(dev_set, last_prompt_hash=prompt_hash)
        except Exception as exc:
            log.exception("evaluation crashed at iter %d: %s", i, exc)
            patch.revert()
            history.append({"iter": i, "kind": "proposal", "kept": False,
                            "reason": f"evaluation_crash:{exc}",
                            "hypothesis": hypothesis})
            continue
        score = res["flip_rate"]
        prompt_hash = res["prompt_hash"]

        kept = score >= baseline - ratchet_slack
        if kept:
            baseline = max(baseline, score)
            last_score = score
            # v5 now reflects this iteration's output; snapshot it as the new
            # best so future reverts restore to here, not the old baseline.
            _snapshot_v5(snapshot_dir)
        else:
            patch.revert()
            # Restore v5's live state to the previous kept iteration.
            _restore_v5(snapshot_dir)

        history.append({
            "iter": i, "kind": "proposal", "kept": kept,
            "score": score, "baseline_after": baseline,
            "hypothesis": hypothesis, "paradigm_shift": paradigm_shift,
            "per_tid": res["per_tid_flipped"],
            "patch_fingerprint": patch.fingerprint(),
            "edited_paths": [e.path for e in patch.edits],
            "wall_clock_s": res["wall_clock_s"],
        })
        log.info("iter %d: score=%.3f kept=%s hypothesis=%s",
                 i, score, kept, hypothesis[:120])

        # Overfit watchdog
        if test_set and i % watchdog_every == 0:
            test_res = evaluate(test_set, last_prompt_hash=prompt_hash,
                                parallel_workers=8)
            history.append({"iter": i, "kind": "watchdog",
                             "dev_score": baseline,
                             "test_score": test_res["flip_rate"],
                             "test_per_tid": test_res["per_tid_flipped"]})
            if (baseline - test_res["flip_rate"]) > overfit_threshold:
                log.warning("OVERFIT detected at iter %d: dev=%.3f test=%.3f; halting",
                            i, baseline, test_res["flip_rate"])
                break

    # ---- final test ----
    final_test = None
    if test_set:
        tr = evaluate(test_set, last_prompt_hash=prompt_hash, parallel_workers=8)
        final_test = tr["flip_rate"]
        history.append({"iter": "final", "kind": "final_test",
                         "test_score": final_test,
                         "test_per_tid": tr["per_tid_flipped"]})

    result = {
        "run_id": run_id,
        "dev_baseline": history.all()[0].get("score"),
        "dev_best": baseline,
        "test_final": final_test,
        "iterations": len(history.all()) - 1,
        "wall_clock_total_s": time.time() - t0,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    log.info("FINAL: %s", result)
    return result
