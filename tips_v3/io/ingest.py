"""Stage 1: read + normalize OWL trajectories from validation_fresh/."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from tips_v3.config import INPUT_DIR, GEMMA_MODEL_ID
from tips_v3.io.schema import Node, Trajectory

log = logging.getLogger(__name__)


def _classify_action(step: dict) -> tuple[str, str | None, dict | None]:
    tool_calls = step.get("tool_calls") or []
    if tool_calls:
        tc = tool_calls[0]
        return "tool_call", tc.get("tool"), tc.get("arguments")
    if step.get("tool_results"):
        return "tool_result", None, None
    content = (step.get("content") or "").strip()
    if "FINAL ANSWER" in content.upper():
        return "final_answer", None, None
    return "reasoning", None, None


def _extract_final_answer(failure_log: list[dict]) -> str:
    for step in reversed(failure_log):
        content = step.get("content") or ""
        upper = content.upper()
        idx = upper.find("FINAL ANSWER")
        if idx == -1:
            continue
        tail = content[idx + len("FINAL ANSWER"):].lstrip(" :\t\n")
        return tail.split("\n", 1)[0].strip()
    return ""


def _parse(raw: dict, path: Path) -> Trajectory | None:
    failure_log = raw.get("failure_log") or []
    if not failure_log:
        log.warning("skip %s: empty failure_log", path.name)
        return None

    oracle = str(raw.get("ground_truth") or "").strip()
    if not oracle:
        log.warning("skip %s: missing ground_truth", path.name)
        return None

    final_answer = _extract_final_answer(failure_log)
    if not final_answer:
        log.warning("skip %s: no FINAL ANSWER in trajectory", path.name)
        return None

    if final_answer.strip().lower() == oracle.strip().lower():
        log.info("skip %s: final answer matches oracle (not a failure)", path.name)
        return None

    nodes: list[Node] = []
    for i, step in enumerate(failure_log):
        action_type, tool_name, tool_args = _classify_action(step)
        agent = step.get("agent") or ""
        level = "planner" if agent.lower() in {"planner", "user"} else "worker"
        obs = None
        tr = step.get("tool_results")
        if tr:
            obs = json.dumps(tr[0], ensure_ascii=False) if isinstance(tr[0], dict) else str(tr[0])
        nodes.append(
            Node(
                step_id=str(step.get("step", i)),
                level=level,
                role=agent,
                action_type=action_type,
                action_content=step.get("content") or "",
                tool_name=tool_name,
                tool_args=tool_args,
                observation=obs,
            )
        )

    tid = str(raw.get("instance_id") or path.stem)
    return Trajectory(
        trajectory_id=tid,
        gaia_task_id=tid,
        query=raw.get("query") or "",
        oracle_answer=oracle,
        agent_final_answer=final_answer,
        agent_model=(raw.get("agentic_system") or {}).get("base_model") or GEMMA_MODEL_ID,
        nodes=nodes,
        raw=raw,
    )


def iter_trajectories(input_dir: Path = INPUT_DIR) -> Iterator[Trajectory]:
    input_dir = Path(input_dir)
    for path in sorted(input_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except Exception as exc:
            log.error("parse error %s: %s", path, exc)
            continue
        traj = _parse(raw, path)
        if traj is not None:
            yield traj


def load_trajectory(trajectory_id: str, input_dir: Path = INPUT_DIR) -> Trajectory | None:
    for traj in iter_trajectories(input_dir):
        if traj.trajectory_id == trajectory_id:
            return traj
    return None
