import numpy as np
import os
"""Identify errors in failed GAIA trajectories, find critical nodes, fix, and rerun.

Same pipeline as graph_identify_and_fix.py but WITHOUT graph data.
Critical nodes are determined by linear ordering: in a sequential trajectory,
an error node is critical if no earlier step is also an error node.

Phase 1: Identify ALL error nodes (LLM, no ground truth).
Phase 2: Find critical nodes (linear — earliest error nodes with no prior error).
Phase 3: Generate fixes for each critical node (LLM replay).
Phase 4: Rerun Magentic-One from after the last critical node.
Phase 5: Score results.
"""

import argparse
import asyncio
import json
import logging
import threading
import re
import time
from pathlib import Path

import httpx

from trajectory.evaluator import evaluate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR = "results/magentic_one/validation"
DEFAULT_OUTPUT_DIR = "error_analysis"
DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it"
MAX_TOKENS = 32768
TEMPERATURE = 0
MAX_RETRIES = 3
BACKOFF_BASE = 2
STEP_TRUNCATE_LIMIT = 2000
STEP_TRUNCATE_HEAD = 1000
STEP_TRUNCATE_TAIL = 500
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TURNS = 50

# ---------------------------------------------------------------------------
# LLM utilities
# ---------------------------------------------------------------------------

def _auth_headers(api_key):
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def call_llm(api_url, model, system, user, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, api_key=None):
    with httpx.Client(timeout=600, headers=_auth_headers(api_key)) as client:
        resp = client.post(api_url, json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature,
        })
        resp.raise_for_status()
        return resp.json()


def call_llm_raw(api_url, model, system, user, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, api_key=None):
    with httpx.Client(timeout=600, headers=_auth_headers(api_key)) as client:
        resp = client.post(api_url, json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature,
        })
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        raw = data["choices"][0]["message"]["content"]
        return raw, {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def call_llm_with_retries(api_url, model, system, user, label, api_key=None):
    raw_response = None
    input_tokens = output_tokens = 0
    latency = 0.0
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = call_llm(api_url, model, system, user, api_key=api_key)
            latency = time.time() - t0
            usage = resp.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            raw_response = resp["choices"][0]["message"]["content"]
            parsed = extract_json(raw_response)
            if parsed is not None:
                return parsed, raw_response, {
                    "input_tokens": input_tokens, "output_tokens": output_tokens,
                    "latency_seconds": round(latency, 2),
                }
            log.warning("%s: JSON parse failed (attempt %d)", label, attempt + 1)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429,) or e.response.status_code >= 500:
                time.sleep(BACKOFF_BASE ** attempt)
            else:
                break
        except Exception as e:
            log.error("%s: %s", label, e)
            time.sleep(BACKOFF_BASE ** attempt)
    return None, raw_response, {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "latency_seconds": round(latency, 2),
    }


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def format_step_content(content):
    if len(content) > STEP_TRUNCATE_LIMIT:
        return content[:STEP_TRUNCATE_HEAD] + "\n...[truncated]...\n" + content[-STEP_TRUNCATE_TAIL:]
    return content


def format_trajectory(steps, truncate=True):
    lines = []
    for s in steps:
        content = format_step_content(s["content"]) if truncate else s["content"]
        lines.append(f"[Step {s['step']}] {s['agent']}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_failed_trajectories(input_dir):
    failed = []
    for fpath in sorted(input_dir.glob("*.json")):
        if fpath.stem == "summary":
            continue
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            continue
        if data.get("task_outcome") != "failure":
            continue
        steps = [s for s in data.get("failure_log", []) if s["agent"] != "system"]
        for i, s in enumerate(steps):
            s["step"] = i
        agent_prompts = {}
        for a in data.get("agentic_system", {}).get("agents", []):
            agent_prompts[a["name"]] = a.get("system_prompt", "")
        failed.append({
            "filename": fpath.name,
            "instance_id": data.get("instance_id", ""),
            "query": data.get("query", ""),
            "ground_truth": data.get("ground_truth", ""),
            "final_answer": data.get("final_answer"),
            "steps": steps,
            "agent_prompts": agent_prompts,
        })
    return failed


# ---------------------------------------------------------------------------
# Phase 1: Identify ALL error nodes
# ---------------------------------------------------------------------------

PHASE1_SYSTEM = (
    "You are analyzing a multi-agent trajectory that produced the wrong answer. "
    "You are given the CORRECT answer for reference. Your task is counterfactual: "
    "identify every step where, had the agent made a different choice, the "
    "trajectory plausibly could have reached the correct answer. This includes "
    "upstream decision points (plans, tool choices, queries) whose own content "
    "may look fine in isolation. Return structured JSON only."
)

_PHASE1_COUNTERFACTUAL_BLOCK = """\
COUNTERFACTUAL TEST: For each step, ask "if this agent had made a better
choice here, does the trajectory plausibly reach the correct answer?" If yes,
flag the step as an error.

This covers:
- REASONING: logical/mathematical mistake, wrong inference
- INFORMATION: misread, hallucinated, or ignored retrieved info
- SEARCH: wrong query, missed search, bad URL
- PLANNING: orchestrator delegated wrong subtask or wrong agent
- TOOL: wrong tool, wrong arguments, misuse
- PREMATURE_TERMINATION: stopped before answer was reachable

Upstream causes count. Flag a step even if its own content looks correct in
isolation, so long as a different choice there would have routed the
trajectory toward the right answer. Flag BOTH decision points (plans,
delegations, queries) AND their downstream consequences when each contributed
independently."""

PHASE1_USER = """\
TASK: Identify ALL steps in this trajectory where a different agent choice
would have put the trajectory on a path toward the correct answer.

The agents were trying to answer this question:
{query}

The agents produced this (WRONG) answer: {final_answer}

THE CORRECT ANSWER IS: {ground_truth}

""" + _PHASE1_COUNTERFACTUAL_BLOCK + """

FULL TRAJECTORY:
{trajectory}

OUTPUT FORMAT:
Return a single JSON object:
{{
  "error_nodes": [
    {{"step": <int>, "agent": "<name>", "error_type": "<type>"}},
    ...
  ]
}}

All step values must be valid indices in [0, {max_step}].
Output only the JSON object."""

PHASE1_CHUNK_USER = """\
TASK: Identify ALL steps IN THE GIVEN WINDOW where a different agent choice
would have put the trajectory on a path toward the correct answer.

The agents were trying to answer this question:
{query}

The agents produced this (WRONG) answer: {final_answer}

THE CORRECT ANSWER IS: {ground_truth}

You are analyzing STEPS {chunk_start}-{chunk_end} of a {total_steps}-step
trajectory that ultimately produced the wrong answer. Flag errors only within
this window.

""" + _PHASE1_COUNTERFACTUAL_BLOCK + """

TRAJECTORY WINDOW (steps {chunk_start}-{chunk_end}):
{trajectory}

OUTPUT FORMAT:
Return a single JSON object:
{{
  "error_nodes": [
    {{"step": <int>, "agent": "<name>", "error_type": "<type>"}},
    ...
  ]
}}

All step values must satisfy {chunk_start} <= step <= {chunk_end}.
Output only the JSON object."""


PHASE1_WINDOW_SIZE = 30
PHASE1_WINDOW_THRESHOLD = 40  # trajectories longer than this are chunked


def _phase1_chunks(steps, window=PHASE1_WINDOW_SIZE):
    """Non-overlapping windows of at most `window` steps."""
    chunks = []
    for start in range(0, len(steps), window):
        chunk = steps[start:start + window]
        if not chunk:
            continue
        chunks.append({
            "start": chunk[0]["step"],
            "end": chunk[-1]["step"],
            "steps": chunk,
        })
    return chunks


def _phase1_single_call(problem_id, traj, api_url, model, api_key=None):
    """One full-trajectory Phase 1 call. Returns (parsed, raw, usage)."""
    trajectory_text = format_trajectory(traj["steps"], truncate=False)
    max_step = len(traj["steps"]) - 1
    user_prompt = PHASE1_USER.format(
        query=traj["query"],
        final_answer=traj["final_answer"] or "(no answer produced)",
        ground_truth=traj["ground_truth"],
        trajectory=trajectory_text,
        max_step=max_step,
    )
    return call_llm_with_retries(api_url, model, PHASE1_SYSTEM, user_prompt, problem_id, api_key=api_key)


def _phase1_chunked_call(problem_id, traj, api_url, model, api_key=None):
    """Sliding-window Phase 1. Returns (all_nodes, raw_responses, usage)."""
    chunks = _phase1_chunks(traj["steps"])
    total = len(traj["steps"])
    all_nodes = []
    raws = []
    agg_usage = {"input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0}
    for ci, chunk in enumerate(chunks):
        chunk_label = f"{problem_id}/chunk{ci+1}of{len(chunks)}[{chunk['start']}-{chunk['end']}]"
        log.info("  chunk %d/%d: steps %d-%d (%d steps)",
                 ci + 1, len(chunks), chunk["start"], chunk["end"], len(chunk["steps"]))
        chunk_text = format_trajectory(chunk["steps"], truncate=False)
        user_prompt = PHASE1_CHUNK_USER.format(
            query=traj["query"],
            final_answer=traj["final_answer"] or "(no answer produced)",
            ground_truth=traj["ground_truth"],
            chunk_start=chunk["start"],
            chunk_end=chunk["end"],
            total_steps=total,
            trajectory=chunk_text,
        )
        parsed, raw, usage = call_llm_with_retries(
            api_url, model, PHASE1_SYSTEM, user_prompt, chunk_label, api_key=api_key,
        )
        raws.append(raw or "")
        agg_usage["input_tokens"] += usage.get("input_tokens", 0)
        agg_usage["output_tokens"] += usage.get("output_tokens", 0)
        agg_usage["latency_seconds"] += usage.get("latency_seconds", 0.0)
        if parsed is None:
            log.warning("    chunk parse failed")
            continue
        for n in parsed.get("error_nodes", []):
            if isinstance(n.get("step"), int) and chunk["start"] <= n["step"] <= chunk["end"]:
                all_nodes.append(n)
    agg_usage["latency_seconds"] = round(agg_usage["latency_seconds"], 2)
    return all_nodes, raws, agg_usage


def run_phase1(trajectories, api_url, model, api_key=None):
    analyses = []
    for idx, traj in enumerate(trajectories):
        problem_id = f"problem_{idx + 1:02d}"
        n_steps = len(traj["steps"])
        log.info("Phase 1: %s (%s) n_steps=%d", problem_id, traj["instance_id"], n_steps)

        entry = {
            "problem_id": problem_id,
            "instance_id": traj["instance_id"],
            "filename": traj["filename"],
            "query": traj["query"],
            "ground_truth": traj["ground_truth"],
            "original_answer": traj["final_answer"],
            "num_steps": n_steps,
        }

        if n_steps > PHASE1_WINDOW_THRESHOLD:
            log.info("  using sliding-window (threshold=%d, window=%d)",
                     PHASE1_WINDOW_THRESHOLD, PHASE1_WINDOW_SIZE)
            raw_nodes, raws, usage = _phase1_chunked_call(problem_id, traj, api_url, model, api_key=api_key)
            entry["raw_llm_response"] = "\n\n---CHUNK---\n\n".join(raws)
            entry.update(usage)
            entry["chunked"] = True
            status = "success" if raw_nodes else "failed"
        else:
            parsed, raw, usage = _phase1_single_call(problem_id, traj, api_url, model, api_key=api_key)
            entry["raw_llm_response"] = raw or ""
            entry.update(usage)
            entry["chunked"] = False
            if parsed is None:
                raw_nodes = []
                status = "failed"
            else:
                max_step = n_steps - 1
                raw_nodes = [n for n in parsed.get("error_nodes", [])
                             if isinstance(n.get("step"), int) and 0 <= n["step"] <= max_step]
                status = "success"

        seen_steps = set()
        deduped = []
        for n in raw_nodes:
            if n["step"] not in seen_steps:
                seen_steps.add(n["step"])
                deduped.append(n)
        entry["error_nodes"] = deduped
        entry["status"] = status
        log.info("  -> %d error nodes: %s", len(deduped), [n["step"] for n in deduped])
        analyses.append(entry)

    return {"analyses": analyses}


# ---------------------------------------------------------------------------
# Phase 2: Find critical nodes (linear — no graph)
# ---------------------------------------------------------------------------

def find_critical_nodes_linear(error_steps):
    """Without graph data, treat trajectory as sequential: every step depends
    on all earlier steps. A critical node is an error node with no earlier
    error node (i.e., no error-node ancestor in the linear chain).

    This means only the earliest error node is critical — all later error
    nodes have it as an ancestor."""
    if not error_steps:
        return []
    earliest = min(error_steps)
    return [earliest]


def run_phase2(phase1_data):
    results = []
    for analysis in phase1_data["analyses"]:
        problem_id = analysis["problem_id"]
        error_nodes = analysis.get("error_nodes", [])
        error_steps = [n["step"] for n in error_nodes]

        critical = find_critical_nodes_linear(error_steps)
        error_by_step = {n["step"]: n for n in error_nodes}
        critical_details = [error_by_step[s] for s in critical]
        last_critical = max(critical) if critical else None

        results.append({
            "problem_id": problem_id,
            "instance_id": analysis["instance_id"],
            "filename": analysis["filename"],
            "all_error_steps": error_steps,
            "critical_nodes": critical_details,
            "critical_steps": critical,
            "last_critical_step": last_critical,
            "status": "success" if critical else "no_errors",
        })
        log.info("%s: %d errors -> %d critical nodes: %s",
                 problem_id, len(error_steps), len(critical), critical)

    return {"results": results}


# ---------------------------------------------------------------------------
# Phase 3: Generate fixes for each critical node
# ---------------------------------------------------------------------------

FIX_SYSTEM = "You are {agent_name}. {agent_system_prompt}"

FIX_USER = """\
CONVERSATION SO FAR:
{prior_context}

YOUR PREVIOUS RESPONSE AT THIS STEP WAS:
{original_content}

THIS STEP WAS FLAGGED AS A {error_type} ERROR.

THE WRONG FINAL ANSWER THE TRAJECTORY PRODUCED: {wrong_final_answer}

Produce a corrected response. You are acting AS the agent at this step.
You must stay in character and produce ONLY what this agent would output
if it had performed correctly.

CRITICAL RULES:
- You are {agent_name}. You can ONLY perform actions available to your role:
  * Orchestrator: delegate tasks to other agents (1-3 sentences).
  * WebSurfer: perform ONE web action (visit_url, web_search, click, etc.).
  * Coder: write a code block for the Executor to run.
  * FileSurfer: perform ONE file operation.
  * Executor: return execution output.
- IMPORTANT: Do NOT include the final answer directly in your response.
  Your correction should fix the PROCESS (plan, action, query, click, search query, tool call,
  reasoning step) so that subsequent steps can arrive at the correct
  answer naturally. The agent at this step would not know the final answer.
- Do NOT derive, compute, or state the final answer in this step.
- Do NOT include information that this agent could not have known at this
  point in the conversation. Use only information available in the
  conversation so far.
- Do NOT skip ahead — do not compress multiple agent actions into one step.
  Produce exactly ONE action that this agent would take.
- Match the EXACT style and length of the original response.
  If the original was {original_length} characters, your correction should be
  similar in length — not longer.
Output ONLY the corrected agent response, no preamble."""



def run_phase3(trajectories, phase2_data, api_url, model):
    traj_by_id = {t["instance_id"]: t for t in trajectories}
    fixes = []

    for result in phase2_data["results"]:
        if result.get("status") != "success" or not result.get("critical_nodes"):
            fixes.append({
                "problem_id": result["problem_id"],
                "instance_id": result["instance_id"],
                "fixes": [], "status": "skipped",
            })
            continue

        problem_id = result["problem_id"]
        instance_id = result["instance_id"]
        traj = traj_by_id[instance_id]
        steps = traj["steps"]

        applied_fixes = {}
        node_fixes = []

        for node in result["critical_nodes"]:
            step_idx = node["step"]
            agent = node["agent"]
            log.info("Phase 3: %s step %d (%s)", problem_id, step_idx, agent)

            agent_system_prompt = traj["agent_prompts"].get(agent, "")

            prior_steps = []
            for s in steps:
                if s["step"] >= step_idx:
                    break
                new_s = dict(s)
                if new_s["step"] in applied_fixes:
                    new_s["content"] = applied_fixes[new_s["step"]]
                prior_steps.append(new_s)

            prior_context = format_trajectory(prior_steps)
            step_data = next(s for s in steps if s["step"] == step_idx)
            original_content = format_step_content(step_data["content"])

            system_prompt = FIX_SYSTEM.format(agent_name=agent, agent_system_prompt=agent_system_prompt)
            user_prompt = FIX_USER.format(
                prior_context=prior_context,
                original_content=original_content,
                error_type=node.get("error_type", "UNKNOWN"),
                wrong_final_answer=traj.get("final_answer") or "(no answer produced)",
                agent_name=agent,
                original_length=len(step_data["content"]),
            )

            fix_max_tokens = min(MAX_TOKENS, max(512, len(step_data["content"]) * 2))

            try:
                t0 = time.time()
                raw_fix, usage = call_llm_raw(api_url, model, system_prompt, user_prompt,
                                              max_tokens=fix_max_tokens)
                latency = time.time() - t0
                corrected = raw_fix.strip()
                applied_fixes[step_idx] = corrected
                node_fixes.append({
                    "step": step_idx, "agent": agent,
                    "error_type": node["error_type"],
                    "corrected_content": corrected,
                    "latency_seconds": round(latency, 2),
                    **usage, "status": "success",
                })
                log.info("  -> fix generated (%d chars)", len(corrected))
            except Exception as e:
                log.error("  -> fix failed: %s", e)
                node_fixes.append({
                    "step": step_idx, "agent": agent,
                    "status": "failed", "error": str(e),
                })

        fixes.append({
            "problem_id": problem_id,
            "instance_id": instance_id,
            "fixes": node_fixes, "status": "success",
        })

    return {"all_fixes": fixes}


# ---------------------------------------------------------------------------
# Phase 4: Rerun with Magentic-One
# ---------------------------------------------------------------------------

def _build_history_messages(steps, critical_fixes, last_critical_step):
    from autogen_agentchat.messages import TextMessage as AutogenTextMessage
    fix_map = {f["step"]: f["corrected_content"] for f in critical_fixes if f.get("status") == "success"}
    messages = []
    for s in steps:
        if s["step"] > last_critical_step:
            break
        content = fix_map.get(s["step"], s["content"])
        messages.append(AutogenTextMessage(content=content, source=s["agent"]))
    return messages


def _find_checkpoint(checkpoint_dir: Path, target_step: int) -> Path | None:
    """Find nearest-descendant checkpoint at index >= target_step (post-fix
    state, so the fixed step is present in message_thread and patchable).
    Falls back to highest available if none >= target. Returns None if dir
    missing or empty."""
    if not checkpoint_dir or not Path(checkpoint_dir).exists():
        return None
    candidates = []
    for f in Path(checkpoint_dir).glob("step_*.json"):
        try:
            idx = int(f.stem.split("_", 1)[1])
        except ValueError:
            continue
        candidates.append((idx, f))
    if not candidates:
        return None
    candidates.sort()
    for idx, f in candidates:
        if idx >= target_step:
            return f
    # No checkpoint >= target: return the highest available as a best-effort
    return candidates[-1][1]


def _build_correction_message(fixes: list) -> str:
    """Build a user-role correction message that lists the fixed step contents."""
    lines = [
        "CORRECTION TO PRIOR CONVERSATION:",
        "The following step(s) above contain errors. Treat the conversation",
        "history as if these agents had instead produced the content shown below,",
        "replacing their original messages:",
        "",
    ]
    for f in sorted(fixes, key=lambda x: x["step"]):
        lines.append(f"--- Step {f['step']} ({f['agent']}) should have been ---")
        lines.append(f["corrected_content"])
        lines.append("")
    lines.append(
        "Continue executing the current plan from this point using the above "
        "corrections as the authoritative content of those earlier steps. "
        "Do not restart planning."
    )
    return "\n".join(lines)


def _normalize_for_match(s: str) -> str:
    """Relax quoting / escaping so tool-call JSON and natural-language
    synthesis messages share comparable substrings."""
    if not isinstance(s, str):
        s = str(s)
    # Strip common JSON escape artifacts
    out = s.replace('\\"', '"').replace("\\'", "'").replace("\\n", " ")
    # Collapse whitespace and lowercase
    return " ".join(out.lower().split())


def _apply_fixes_to_state(state: dict, fixes: list) -> tuple[dict, int, list]:
    """Mutate a team save_state dict so fixed step contents replace originals.

    For each fix (in ascending step order): find the first unpatched
    message in the Orchestrator's message_thread whose source matches the
    fix's agent and whose normalized content shares a 40-char substring
    with the fix's normalized original_content. Replace that message's
    content with corrected_content.

    After patching, clear orchestrator `plan` / `facts` / turn counters so
    they re-derive from the edited thread on resume.

    Returns (patched_state, n_patched, unpatched_fixes).
    """
    agent_states = state.get("agent_states", {})
    orch = agent_states.get("MagenticOneOrchestrator", {})
    thread = orch.get("message_thread", [])
    n_patched = 0
    matched_indices: set[int] = set()
    unpatched = []

    for f in sorted(fixes, key=lambda x: x["step"]):
        agent = f.get("agent")
        original = f.get("original_content", "") or ""
        corrected = f.get("corrected_content", "") or ""
        if not agent or not original or not corrected:
            unpatched.append(f)
            continue

        orig_norm = _normalize_for_match(original)
        if len(orig_norm) < 20:
            unpatched.append(f)
            continue

        # Pick a few distinctive substrings from the normalized original
        windows = []
        step = max(1, len(orig_norm) // 8)
        for start in range(0, max(1, len(orig_norm) - 40), step):
            windows.append(orig_norm[start:start + 40])
        if not windows:
            windows.append(orig_norm[:40])

        found_idx = None
        for i, msg in enumerate(thread):
            if i in matched_indices:
                continue
            if msg.get("source") != agent:
                continue
            target_norm = _normalize_for_match(msg.get("content", ""))
            if any(w in target_norm for w in windows):
                found_idx = i
                break

        if found_idx is not None:
            # Replace content; force type=TextMessage since corrected is a
            # string (MultiModalMessage expects a list — keeping original
            # type fails state schema validation).
            thread[found_idx]["content"] = corrected
            thread[found_idx]["type"] = "TextMessage"
            matched_indices.add(found_idx)
            n_patched += 1
        else:
            thread.append({
                "source": agent,
                "type": "TextMessage",
                "content": corrected,
                "models_usage": None,
                "metadata": {},
            })
            n_patched += 1
            unpatched.append(f)

    # Clear derived orchestrator fields so plan/facts re-derive from the
    # patched message_thread on resume rather than citing pre-patch content.
    if n_patched > 0:
        orch["plan"] = ""
        orch["facts"] = ""
        orch["n_rounds"] = 0
        orch["n_stalls"] = 0

    return state, n_patched, unpatched


async def _rerun_single(task, history_messages, runner, writer, problem_id="unknown",
                         checkpoint_dir=None, fixes=None, last_fixed_step=None):
    """Rerun the trajectory.

    If `checkpoint_dir` is provided and a checkpoint exists at or before
    `last_fixed_step`, load the team state from the checkpoint and pass the
    correction message (built from `fixes`) as the continuation task — this
    preserves the orchestrator's ledger and suppresses the fresh re-plan.

    Otherwise fall back to the legacy text-replay path (`history_messages` as
    task input).
    """
    from runners.magentic_one_runner import (
        MessageLogger, AgentEventHandler, _parse_args, GAIA_FINAL_ANSWER_PROMPT,
    )
    from autogen_agentchat.agents import CodeExecutorAgent
    from autogen_agentchat.messages import (
        ToolCallExecutionEvent, ToolCallRequestEvent,
        ToolCallSummaryMessage, TextMessage, ThoughtEvent,
    )
    from autogen_agentchat.teams import MagenticOneGroupChat
    from autogen_ext.agents.file_surfer import FileSurfer
    from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
    from autogen_ext.agents.web_surfer import MultimodalWebSurfer
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    import tempfile, shutil

    task_id = task["task_id"]
    msg_logger = MessageLogger(f"{problem_id}_rerun")

    # Same agent names as MagenticOneRunner so load_state is compatible:
    # Assistant (MagenticOneCoderAgent), ComputerTerminal, FileSurfer, WebSurfer.
    work_dir = Path(tempfile.mkdtemp(prefix=f"rerun_{task_id[:8]}_"))

    # Stage referenced file so FileSurfer/Coder can find it (matches original runner)
    filename = task.get("file_name", "") or ""
    if filename:
        gaia_files = getattr(runner, "gaia_files_dir", None)
        if gaia_files is not None:
            src = Path(gaia_files) / task.get("file_path", filename)
            if src.exists():
                shutil.copy2(src, work_dir / filename)

    coder = MagenticOneCoderAgent("Assistant", model_client=runner.client)
    code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
    executor = CodeExecutorAgent("ComputerTerminal", code_executor=code_executor)
    file_surfer = FileSurfer(name="FileSurfer", model_client=runner.client)
    web_surfer = MultimodalWebSurfer(
        name="WebSurfer", model_client=runner.client,
        downloads_folder=str(work_dir), to_save_screenshots=False, headless=True,
    )

    final_answer_prompt = GAIA_FINAL_ANSWER_PROMPT.format(prompt=task["Question"])
    team = MagenticOneGroupChat(
        [coder, executor, file_surfer, web_surfer],
        model_client=runner.client, max_turns=DEFAULT_MAX_TURNS,
        final_answer_prompt=final_answer_prompt,
    )

    # Decide rerun mode: state-restore vs text-replay
    cp_path = None
    if checkpoint_dir is not None and last_fixed_step is not None:
        cp_path = _find_checkpoint(Path(checkpoint_dir), last_fixed_step)

    run_mode = "state_restore" if cp_path is not None else "text_replay"
    log.info("  rerun mode: %s%s", run_mode,
             f" (checkpoint={cp_path.name})" if cp_path else "")

    core_logger = logging.getLogger("autogen_core")
    agent_handler = AgentEventHandler(msg_logger, writer)
    core_logger.addHandler(agent_handler)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{problem_id}_rerun.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    core_logger.addHandler(file_handler)
    core_logger.setLevel(logging.INFO)

    # Build task input depending on mode
    if run_mode == "state_restore":
        state = json.loads(cp_path.read_text())
        state, n_patched, appended = _apply_fixes_to_state(state, fixes or [])
        log.info("  state-edit: %d patched in-place (%d replaced, %d appended as synthetic)",
                 n_patched, n_patched - len(appended), len(appended))
        await team.load_state(state)
        stream_task = "continue"
    else:
        stream_task = history_messages

    final_answer = None
    try:
        async for event in team.run_stream(task=stream_task):
            agent_name = str(getattr(event, "source", "unknown"))
            if isinstance(event, ToolCallRequestEvent):
                tool_calls = [{"tool": fc.name, "arguments": _parse_args(fc.arguments)} for fc in event.content]
                content = "; ".join(f"{tc['tool']}({json.dumps(tc['arguments'])})" for tc in tool_calls)
                writer.add_step(agent=agent_name, content=content, tool_calls=tool_calls)
            elif isinstance(event, ToolCallExecutionEvent):
                for result in event.content:
                    rc = result.content if result.content else ""
                    ie = getattr(result, "is_error", False)
                    writer.add_step(agent=agent_name,
                                    content=f"[{result.name}] {'ERROR: ' if ie else ''}{rc}",
                                    tool_results=[{"tool": result.name, "output": rc, "is_error": ie}])
            elif isinstance(event, ToolCallSummaryMessage):
                c = event.content if isinstance(event.content, str) else str(event.content)
                writer.add_step(agent=agent_name, content=c)
            elif isinstance(event, TextMessage):
                tc_parsed = []
                if agent_name == "WebSurfer":
                    tc_match = re.match(r'^(\w+)\(\s*(\{.*\})\s*\)$', event.content.strip(), re.DOTALL)
                    if tc_match:
                        try:
                            tc_parsed = [{"tool": tc_match.group(1),
                                          "arguments": json.loads(tc_match.group(2))}]
                        except json.JSONDecodeError:
                            pass
                writer.add_step(agent=agent_name, content=event.content,
                                tool_calls=tc_parsed or None)
            elif isinstance(event, ThoughtEvent):
                pass
            elif hasattr(event, "messages"):
                if event.messages:
                    last_msg = event.messages[-1]
                    lc = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                    final_answer = await runner._extract_answer(lc, task["Question"])
                writer.add_step(agent="system",
                                content=f"Task completed. Stop reason: {getattr(event, 'stop_reason', 'unknown')}")
            elif hasattr(event, "source") and hasattr(event, "content"):
                c = event.content if isinstance(event.content, str) else str(event.content)
                lt, la = agent_handler.pop_last_tool()
                if lt and agent_name == "WebSurfer":
                    writer.add_step(agent=agent_name, content=c,
                                    tool_results=[{"tool": lt, "output": c, "is_error": False}])
                else:
                    writer.add_step(agent=agent_name, content=c)
    finally:
        core_logger.removeHandler(agent_handler)
        core_logger.removeHandler(file_handler)
        file_handler.close()
        await team.reset()
    return final_answer


async def _run_phase4_async(trajectories, phase2_data, phase3_data,
                            output_dir, model, timeout):
    from runners.magentic_one_runner import MagenticOneRunner, AGENT_DEFINITIONS
    from trajectory.writer import TrajectoryWriter

    traj_by_id = {t["instance_id"]: t for t in trajectories}
    fixes_by_id = {f["instance_id"]: f for f in phase3_data["all_fixes"]}

    data_dir = Path("dataset/gaia")
    tasks_list = []
    tasks_by_id = {}
    for line in (data_dir / "validation.jsonl").read_text().splitlines():
        task = json.loads(line)
        tasks_list.append(task)
        tasks_by_id[task["task_id"]] = task

    file_to_task_id = {}
    for t in trajectories:
        iid = t["instance_id"]
        stem = t["filename"].replace(".json", "")
        if stem in tasks_by_id:
            file_to_task_id[iid] = stem
        else:
            # Map by index from instance_id (e.g. "gaia_validation_0000" -> tasks_list[0])
            import re as _re
            m = _re.search(r"(\d+)$", iid)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < len(tasks_list):
                    file_to_task_id[iid] = tasks_list[idx]["task_id"]

    runner = MagenticOneRunner(
        data_dir=str(data_dir), output_dir=str(output_dir / "reruns"), model=model,
    )
    rerun_dir = output_dir / "reruns"
    rerun_dir.mkdir(parents=True, exist_ok=True)

    simulations = []

    for result in phase2_data["results"]:
        instance_id = result["instance_id"]
        problem_id = result["problem_id"]
        fix_data = fixes_by_id.get(instance_id, {})

        # Use original problem number from instance_id suffix (e.g., "gaia_validation_0002" -> 2)
        import re as _re
        _m = _re.search(r"(\d+)$", instance_id)
        orig_num = int(_m.group(1)) if _m else int(problem_id.split('_')[1])

        # Skip if rerun output already exists
        rerun_file = rerun_dir / f"problem_{orig_num:02d}.json"
        if rerun_file.exists():
            log.info("Phase 4: %s — skipping (rerun file exists)", problem_id)
            simulations.append({
                "problem_id": problem_id, "instance_id": instance_id,
                "status": "skipped", "reason": "rerun file exists",
            })
            continue

        if result.get("status") != "success" or not result.get("critical_steps"):
            simulations.append({
                "problem_id": problem_id, "instance_id": instance_id,
                "status": "skipped", "reason": "no critical nodes",
            })
            continue

        if fix_data.get("status") != "success":
            simulations.append({
                "problem_id": problem_id, "instance_id": instance_id,
                "status": "skipped", "reason": "fixes failed",
            })
            continue

        traj = traj_by_id[instance_id]
        task_id = file_to_task_id[instance_id]
        task = tasks_by_id.get(task_id)
        if task is None:
            simulations.append({
                "problem_id": problem_id, "instance_id": instance_id,
                "status": "failed", "error": "task_not_found",
            })
            continue

        last_critical = result["last_critical_step"]
        critical_fixes = fix_data["fixes"]

        history_messages = _build_history_messages(traj["steps"], critical_fixes, last_critical)
        log.info("Phase 4: %s — %d history messages (up to step %d, %d fixes applied)",
                 problem_id, len(history_messages), last_critical, len(critical_fixes))

        writer = TrajectoryWriter(
            task=task, framework="magentic-one-rerun", model=model,
            output_dir=str(rerun_dir), instance_id=instance_id + "_rerun",
            problem_number=orig_num, agents=AGENT_DEFINITIONS,
        )
        writer.enable_live_log()

        entry = {
            "problem_id": problem_id, "instance_id": instance_id,
            "critical_steps": result["critical_steps"],
            "last_critical_step": last_critical,
        }

        try:
            t0 = time.time()
            # Checkpoint dir sits alongside the baseline trajectory files
            cp_dir = Path("results/magentic_one/validation") / "checkpoints" / f"problem_{orig_num:02d}"
            rerun_task = asyncio.ensure_future(
                _rerun_single(
                    task, history_messages, runner, writer, problem_id=problem_id,
                    checkpoint_dir=cp_dir, fixes=critical_fixes, last_fixed_step=last_critical,
                ),
            )
            loop = asyncio.get_event_loop()
            watchdog = threading.Timer(
                timeout,
                lambda: loop.call_soon_threadsafe(rerun_task.cancel),
            )
            watchdog.start()
            try:
                final_answer = await rerun_task
            finally:
                watchdog.cancel()
            latency = time.time() - t0
            writer.finalize(final_answer=final_answer)
            entry.update({
                "rerun_answer": final_answer,
                "latency_seconds": round(latency, 2),
                "status": "success",
            })
            log.info("  -> rerun answer: %s (%.1fs)", final_answer, latency)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            writer.finalize(error="Timeout")
            entry.update({"status": "failed", "error": "timeout"})
            log.error("  -> timed out after %ds", timeout)
        except Exception as e:
            writer.finalize(error=str(e))
            entry.update({"status": "failed", "error": str(e)})
            log.error("  -> failed: %s", e)

        simulations.append(entry)

    await runner.cleanup()
    return {"simulations": simulations}


def run_phase4(trajectories, phase2_data, phase3_data, output_dir, model, timeout):
    return asyncio.run(_run_phase4_async(
        trajectories, phase2_data, phase3_data, output_dir, model, timeout,
    ))


# ---------------------------------------------------------------------------
# Phase 5: Scoring and text reports
# ---------------------------------------------------------------------------

def run_scoring(phase1_data, phase2_data, phase3_data, phase4_data, output_dir):
    analysis_by_id = {a["instance_id"]: a for a in phase1_data["analyses"]}
    critical_by_id = {r["instance_id"]: r for r in phase2_data["results"]}
    fixes_by_id = {f["instance_id"]: f for f in phase3_data["all_fixes"]}
    sim_by_id = {s["instance_id"]: s for s in phase4_data["simulations"]}

    results = []
    flipped = 0
    total = 0

    for instance_id, analysis in analysis_by_id.items():
        critical = critical_by_id.get(instance_id, {})
        fix_data = fixes_by_id.get(instance_id, {})
        sim = sim_by_id.get(instance_id, {})
        problem_id = analysis["problem_id"]

        ground_truth = analysis["ground_truth"]
        original_answer = analysis.get("original_answer")
        rerun_answer = sim.get("rerun_answer")

        original_correct = evaluate_answer(original_answer, ground_truth)
        rerun_correct = evaluate_answer(rerun_answer, ground_truth) if rerun_answer else False
        is_flipped = not original_correct and rerun_correct
        if is_flipped:
            flipped += 1
        total += 1

        result = {
            "problem_id": problem_id,
            "instance_id": instance_id,
            "ground_truth": ground_truth,
            "original_answer": original_answer,
            "original_correct": original_correct,
            "rerun_answer": rerun_answer,
            "rerun_correct": rerun_correct,
            "flipped_to_correct": is_flipped,
            "all_error_steps": [n["step"] for n in analysis.get("error_nodes", [])],
            "critical_steps": critical.get("critical_steps", []),
            "last_critical_step": critical.get("last_critical_step"),
        }
        results.append(result)
        write_report(result, analysis, critical, fix_data, sim, output_dir)

    summary = {
        "total_failures_analyzed": total,
        "flipped_to_correct": flipped,
        "remained_wrong": total - flipped,
        "original_correct_total": 3,
        "projected_total_correct": 3 + flipped,
        "projected_total_accuracy": round((3 + flipped) / 10, 2),
    }
    return {"results": results, "summary": summary}


def write_report(result, analysis, critical, fix_data, sim, output_dir):
    problem_id = result["problem_id"]
    query = analysis.get("query", "")
    query_display = query[:200] + ("..." if len(query) > 200 else "")

    lines = [
        f"{'=' * 70}",
        f"  {problem_id.upper()} — {result['instance_id']}",
        f"{'=' * 70}",
        "",
        "PROBLEM INFO",
        f"  Query:           {query_display}",
        f"  Ground Truth:    {result['ground_truth']}",
        f"  Original Answer: {result['original_answer']}",
        f"  Original Correct: {result['original_correct']}",
        "",
        f"ALL ERROR NODES: {result['all_error_steps']}",
        f"CRITICAL NODES:  {result['critical_steps']}",
        f"LAST CRITICAL:   {result.get('last_critical_step', 'N/A')}",
        "",
    ]

    for node in critical.get("critical_nodes", []):
        step = node["step"]
        lines.append(f"--- Critical Node: Step {step} ({node['agent']}) ---")
        lines.append(f"  Error Type:      {node['error_type']}")
        fix = next((f for f in fix_data.get("fixes", []) if f.get("step") == step), None)
        if fix and fix.get("status") == "success":
            corrected = fix["corrected_content"]
            if len(corrected) > 500:
                corrected = corrected[:400] + "\n  ...[truncated]..."
            lines.append(f"  Fix:")
            lines.append("    " + "\n    ".join(corrected.split("\n")))
        lines.append("")

    lines.extend([
        "RERUN RESULT",
        f"  Rerun Answer:      {result.get('rerun_answer', 'N/A')}",
        f"  Rerun Correct:     {result.get('rerun_correct', False)}",
        f"  Flipped to Correct: {result.get('flipped_to_correct', False)}",
        "",
    ])

    txt_path = output_dir / f"{problem_id}.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  Written: %s", txt_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def select_fix_targets(trajectory, critical_nodes, fix_target, instance_id, trial_idx=0):
    """
    Implements the target selection logic for the Wrong-Fix Robustness Experiment.
    """
    # For reproducibility: seed based on instance, condition, and trial
    seed = hash((instance_id, fix_target, trial_idx))
    np.random.seed(seed)
    
    all_steps = list(range(len(trajectory)))
    error_steps = [i for i, step in enumerate(trajectory) if step.get('is_error', False)]
    non_error_steps = [i for i, step in enumerate(trajectory) if not step.get('is_error', False)]
    
    if fix_target == 'critical': # C2
        return critical_nodes
    
    elif fix_target == 'random_error': # C3
        eligible = [s for s in error_steps if s not in critical_nodes]
        return [np.random.choice(eligible)] if eligible else []
        
    elif fix_target == 'random_nonerror': # C4
        eligible = [s for s in non_error_steps if s not in critical_nodes]
        return [np.random.choice(eligible)] if eligible else []
        
    elif fix_target == 'adjacent': # C5
        if not critical_nodes: return []
        crit = critical_nodes[0] # Base on first critical node
        eligible = [s for s in [crit-1, crit+1] if 0 <= s < len(trajectory) and s not in critical_nodes]
        return [np.random.choice(eligible)] if eligible else []
        
    elif fix_target == 'all_errors': # C7
        return error_steps
        
    elif fix_target == 'none': # C0
        return []
        
    elif fix_target == 'empty': # C6
        return critical_nodes
        
    return critical_nodes

def main():
    parser = argparse.ArgumentParser(description="Identify errors, fix, and rerun (no graph)")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], default=None)
    parser.add_argument("--api-key", default=None,
                        help="Bearer token for the LLM endpoint (e.g., OpenRouter). "
                             "If omitted, no Authorization header is sent.")
    parser.add_argument('--fix-target',
                    type=str,
                    choices=['critical', 'random_error', 'random_nonerror', 'adjacent', 'empty', 'all_errors', 'none'],
                    default='critical',
                    help='Target for the fix experiment (C0-C7)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p1_path = output_dir / "step1_all_errors.json"
    p2_path = output_dir / "step2_critical_nodes.json"
    p3_path = output_dir / "step3_fixes.json"
    p4_path = output_dir / "step4_reruns.json"
    p5_path = output_dir / "step5_results.json"

    trajectories = load_failed_trajectories(input_dir)
    log.info("Loaded %d failed trajectories", len(trajectories))

    if args.phase is None or args.phase == 1:
        log.info("=== PHASE 1: Identify ALL error nodes ===")
        p1 = run_phase1(trajectories, args.api_url, args.model, api_key=args.api_key)
        p1_path.write_text(json.dumps(p1, indent=2, ensure_ascii=False))

    if args.phase is None or args.phase == 2:
        p1 = json.loads(p1_path.read_text())
        log.info("=== PHASE 2: Find critical nodes (linear) ===")
        p2 = run_phase2(p1)
        p2_path.write_text(json.dumps(p2, indent=2, ensure_ascii=False))

    if args.phase is None or args.phase == 3:
        p2 = json.loads(p2_path.read_text())
        log.info("=== PHASE 3: Generate fixes ===")
        p3 = run_phase3(trajectories, p2, args.api_url, args.model)
        p3_path.write_text(json.dumps(p3, indent=2, ensure_ascii=False))

    if args.phase is None or args.phase == 4:
        p2 = json.loads(p2_path.read_text())
        p3 = json.loads(p3_path.read_text())
        log.info("=== PHASE 4: Rerun with Magentic-One ===")
        p4 = run_phase4(trajectories, p2, p3, output_dir, args.model, args.timeout)
        p4_path.write_text(json.dumps(p4, indent=2, ensure_ascii=False))

    if args.phase is None or args.phase == 5:
        p1 = json.loads(p1_path.read_text())
        p2 = json.loads(p2_path.read_text())
        p3 = json.loads(p3_path.read_text())
        p4 = json.loads(p4_path.read_text())
        log.info("=== PHASE 5: Scoring ===")
        p5 = run_scoring(p1, p2, p3, p4, output_dir)
        p5_path.write_text(json.dumps(p5, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
