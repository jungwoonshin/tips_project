"""Bounded replay: memory-restore into a fresh OWL ChatAgent, splice fixes at M-nodes,
drive forward with the continuation prompt, evaluate the final answer vs oracle.

Wraps (does not rewrite) the memory-restoration pattern from
`owl_counterfactual_rerun.py`. A thin shim is used so unit tests can run without the
CAMEL stack installed — the real adapter is imported lazily.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import json
from dataclasses import dataclass
from typing import Callable, Protocol

from tips_v3.config import REPLAY_CACHE_DB
from tips_v3.io.schema import Fix, ReplayResult, Trajectory

log = logging.getLogger(__name__)


class ReplayBackend(Protocol):
    def run(
        self,
        traj: Trajectory,
        fix_map: dict[str, Fix],
        seed: int,
    ) -> ReplayResult: ...


_FINAL_ANSWER_RE_BACKEND = None  # lazy — owl_counterfactual_rerun imports camel


def _extract_final_if_terminal_fix(fix_map: dict[str, Fix]) -> str | None:
    """If the latest fix (by step) already contains a 'FINAL ANSWER: X' line,
    return X without running Gemma.

    Semantic: when Sonnet's fix concludes the trajectory itself (typical for
    REASONING/PLANNING fixes at terminal steps), running `agent.step` again
    would let Gemma regenerate the answer and erase the fix. Accepting the
    fix's FINAL ANSWER directly is the honest reading of 'what if the agent
    had reasoned this way at this step'."""
    if not fix_map:
        return None
    import re
    try:
        last_step = max(int(nid) for nid in fix_map.keys())
    except ValueError:
        return None
    fix = fix_map.get(str(last_step))
    if fix is None:
        return None
    m = re.search(r"FINAL ANSWER[:\s]+(.+?)(?:\n|$)", fix.new_content, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


@dataclass
class OWLReplayBackend:
    """Real backend: shapes our Trajectory into the dict that
    `owl_counterfactual_rerun.rerun_owl` expects, then runs the replay."""

    api_url: str = "http://localhost:8000/v1/chat/completions"
    model: str = "google/gemma-4-31b-it"
    timeout: int = 300

    def run(self, traj: Trajectory, fix_map: dict[str, Fix], seed: int) -> ReplayResult:
        import asyncio
        from owl_counterfactual_rerun import rerun_owl
        from trajectory.writer import TrajectoryWriter

        if not fix_map:
            return ReplayResult(seed=seed, final_answer="", flipped=False,
                                error="empty_fix_map")

        terminal = _extract_final_if_terminal_fix(fix_map)
        if terminal is not None:
            from tips_v3.eval import gaia_scorer
            flipped = gaia_scorer(terminal, traj.oracle_answer)
            log.info("replay short-circuit: fix at max-step contains FINAL ANSWER=%r (flipped=%s)",
                     terminal[:80], flipped)
            return ReplayResult(seed=seed, final_answer=terminal,
                                flipped=flipped, error=None)

        raw = traj.raw
        steps = [s for s in raw.get("failure_log", []) if s.get("agent") != "system"]
        for i, s in enumerate(steps):
            s["step"] = i
        agent_prompts = {
            a["name"]: a.get("system_prompt", "")
            for a in (raw.get("agentic_system") or {}).get("agents", [])
        }
        traj_dict = {
            "instance_id": traj.trajectory_id,
            "query": traj.query,
            "ground_truth": traj.oracle_answer,
            "final_answer": traj.agent_final_answer,
            "steps": steps,
            "agent_prompts": agent_prompts,
        }

        fixes_list = []
        for nid, fix in fix_map.items():
            try:
                step_idx = int(nid)
            except ValueError:
                continue
            fixes_list.append({"step": step_idx, "corrected_content": fix.new_content})
        if not fixes_list:
            return ReplayResult(seed=seed, final_answer="", flipped=False,
                                error="no_int_step_ids")

        task = {
            "task_id": traj.trajectory_id,
            "Question": traj.query,
            "Final answer": traj.oracle_answer,
        }
        from tips_v3.config import OUTPUT_DIR
        writer_dir = OUTPUT_DIR / "_replay_writers" / traj.trajectory_id
        writer = TrajectoryWriter(
            task=task,
            framework="OWL/Workforce",
            model=self.model,
            output_dir=str(writer_dir),
            instance_id=traj.trajectory_id,
        )

        try:
            result = asyncio.run(
                rerun_owl(traj_dict, fixes_list, task, writer,
                          self.model, self.api_url, self.timeout)
            )
        except Exception as exc:
            log.exception("replay error for %s seed=%d", traj.trajectory_id, seed)
            return ReplayResult(seed=seed, final_answer="", flipped=False, error=str(exc))

        from tips_v3.eval import gaia_scorer
        final = result.get("final_answer") or ""
        flipped = bool(final and gaia_scorer(final, traj.oracle_answer))
        return ReplayResult(seed=seed, final_answer=final, flipped=flipped,
                            error=result.get("error"))


def _fix_map_key(fix_map: dict[str, Fix]) -> str:
    items = sorted(
        (nid, f.modified_field, f.new_content) for nid, f in fix_map.items()
    )
    blob = json.dumps(items, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()


class ReplayCache:
    """sqlite-backed cache keyed by (trajectory_id, fix_map_hash, seed).

    Uses check_same_thread=False + a lock so the cache can be shared across
    ThreadPoolExecutor workers (autoresearch evaluator runs dev trajectories
    in parallel). SQLite3 itself is thread-safe at the library level."""

    def __init__(self, db_path=REPLAY_CACHE_DB):
        import threading
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " tid TEXT, fmkey TEXT, seed INTEGER, final_answer TEXT, flipped INTEGER,"
            " error TEXT, PRIMARY KEY(tid, fmkey, seed))"
        )
        self._conn.commit()

    def get(self, tid: str, fmkey: str, seed: int) -> ReplayResult | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT final_answer, flipped, error FROM cache WHERE tid=? AND fmkey=? AND seed=?",
                (tid, fmkey, seed),
            ).fetchone()
        if row is None:
            return None
        final, flipped, error = row
        return ReplayResult(seed=seed, final_answer=final, flipped=bool(flipped), error=error)

    def put(self, tid: str, fmkey: str, result: ReplayResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, ?)",
                (tid, fmkey, result.seed, result.final_answer, int(result.flipped), result.error),
            )
            self._conn.commit()


class DryRunBackend:
    """Fallback used when the OWL/CAMEL stack isn't installed. Records the
    replay as 'not attempted' so the pipeline still writes difficult records
    with a clear reason instead of crashing."""

    def run(self, traj: Trajectory, fix_map: dict[str, Fix], seed: int) -> ReplayResult:
        return ReplayResult(seed=seed, final_answer="",
                            flipped=False, error="dry_run_backend_no_camel")


def _default_backend() -> ReplayBackend:
    try:
        import camel  # noqa: F401
    except Exception:
        log.warning("camel not installed; using DryRunBackend (no real replay)")
        return DryRunBackend()
    return OWLReplayBackend()


class BoundedReplay:
    def __init__(self, backend: ReplayBackend | None = None, cache: ReplayCache | None = None):
        self.backend = backend or _default_backend()
        self.cache = cache or ReplayCache()

    def run(
        self,
        traj: Trajectory,
        fix_map: dict[str, Fix],
        seeds: list[int],
    ) -> list[ReplayResult]:
        fmkey = _fix_map_key(fix_map)
        results: list[ReplayResult] = []
        for seed in seeds:
            cached = self.cache.get(traj.trajectory_id, fmkey, seed)
            if cached is not None:
                results.append(cached)
                continue
            res = self.backend.run(traj, fix_map, seed)
            self.cache.put(traj.trajectory_id, fmkey, res)
            results.append(res)
        return results

    @staticmethod
    def flip_rate(results: list[ReplayResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.flipped) / len(results)
