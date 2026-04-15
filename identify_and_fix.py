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
DEFAULT_MODEL = "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ"
MAX_TOKENS = 131072
TEMPERATURE = 0
MAX_RETRIES = 3
BACKOFF_BASE = 2
STEP_TRUNCATE_LIMIT = 2000
STEP_TRUNCATE_HEAD = 1000
STEP_TRUNCATE_TAIL = 500
DEFAULT_TIMEOUT = 300

# ---------------------------------------------------------------------------
# LLM utilities
# ---------------------------------------------------------------------------

def call_llm(api_url, model, system, user, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    with httpx.Client(timeout=600) as client:
        resp = client.post(api_url, json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature,
        })
        resp.raise_for_status()
        return resp.json()


def call_llm_raw(api_url, model, system, user, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    with httpx.Client(timeout=600) as client:
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


def call_llm_with_retries(api_url, model, system, user, label):
    raw_response = None
    input_tokens = output_tokens = 0
    latency = 0.0
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = call_llm(api_url, model, system, user)
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
    "Identify ALL steps that contain errors — not just the first one. "
    "You do NOT know the correct answer. Return structured JSON only."
)

PHASE1_USER = """\
TASK: Identify ALL steps in this trajectory where an agent made an error.

The agents were trying to answer this question:
{query}

The agents produced this answer: {final_answer}

FULL TRAJECTORY:
{trajectory}

INSTRUCTIONS:
1. Read through the entire trajectory carefully.
2. Identify EVERY step where the agent made a clear mistake:
   - REASONING: logical or mathematical mistake
   - INFORMATION: misread, hallucinated, or ignored retrieved information
   - SEARCH: searched for wrong thing or failed to search
   - PLANNING: orchestrator assigned wrong subtask or wrong agent
   - TOOL: used wrong tool or used a tool incorrectly
   - PREMATURE_TERMINATION: stopped too early
3. List ALL such error steps, not just the earliest one.

OUTPUT FORMAT:
Return a single JSON object:
{{
  "error_nodes": [
    {{"step": <int>, "agent": "<name>", "error_type": "<type>", "what_went_wrong": "<1-2 sentences>"}},
    ...
  ]
}}

All step values must be valid indices in [0, {max_step}].
Output only the JSON object."""


def run_phase1(trajectories, api_url, model):
    analyses = []
    for idx, traj in enumerate(trajectories):
        problem_id = f"problem_{idx + 1:02d}"
        log.info("Phase 1: %s (%s)", problem_id, traj["instance_id"])

        trajectory_text = format_trajectory(traj["steps"], truncate=False)
        max_step = len(traj["steps"]) - 1

        user_prompt = PHASE1_USER.format(
            query=traj["query"],
            final_answer=traj["final_answer"] or "(no answer produced)",
            trajectory=trajectory_text,
            max_step=max_step,
        )

        parsed, raw, usage = call_llm_with_retries(api_url, model, PHASE1_SYSTEM, user_prompt, problem_id)

        entry = {
            "problem_id": problem_id,
            "instance_id": traj["instance_id"],
            "filename": traj["filename"],
            "query": traj["query"],
            "ground_truth": traj["ground_truth"],
            "original_answer": traj["final_answer"],
            "num_steps": len(traj["steps"]),
            "raw_llm_response": raw or "",
            **usage,
        }

        if parsed is None:
            entry["status"] = "failed"
            entry["error_nodes"] = []
        else:
            raw_nodes = parsed.get("error_nodes", [])
            valid = [n for n in raw_nodes
                     if isinstance(n.get("step"), int) and 0 <= n["step"] <= max_step]
            # Deduplicate by step index, keeping first occurrence
            seen_steps = set()
            deduped = []
            for n in valid:
                if n["step"] not in seen_steps:
                    seen_steps.add(n["step"])
                    deduped.append(n)
            entry["error_nodes"] = deduped
            entry["status"] = "success"
            log.info("  -> %d error nodes: %s", len(valid), [n["step"] for n in valid])

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

THIS RESPONSE WAS WRONG BECAUSE:
{what_went_wrong}

Produce a corrected response. Requirements:
- Fix ONLY the specific mistake described above.
- Match the EXACT style and length of the original response shown above.
  If the original was {original_length} characters, your correction should be
  similar in length — not longer.
- Do NOT add explanations, reasoning, or analysis that the agent would not
  have produced. Just output the corrected agent response, nothing else.
- For Orchestrator: short delegation instructions (1-3 sentences).
- For WebSurfer: a tool call like visit_url(...) or web_search(...).
- For Coder: a code block.
- For FileSurfer: a file operation or extracted content summary.
Output ONLY the corrected response, no preamble."""


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
                what_went_wrong=node["what_went_wrong"],
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
                    "what_went_wrong": node["what_went_wrong"],
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


async def _rerun_single(task, history_messages, runner, writer, problem_id="unknown"):
    from runners.magentic_one_runner import MessageLogger, AgentEventHandler, _parse_args
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
    import tempfile

    task_id = task["task_id"]
    msg_logger = MessageLogger(f"{problem_id}_rerun")

    web_surfer = MultimodalWebSurfer("WebSurfer", model_client=runner.client, headless=True)
    file_surfer = FileSurfer("FileSurfer", model_client=runner.client)
    coder = MagenticOneCoderAgent("Coder", model_client=runner.client)
    work_dir = Path(tempfile.mkdtemp(prefix=f"rerun_{task_id[:8]}_"))
    code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
    executor = CodeExecutorAgent("Executor", code_executor=code_executor)

    team = MagenticOneGroupChat(
        participants=[web_surfer, file_surfer, coder, executor],
        model_client=runner.client, max_turns=100,
    )

    core_logger = logging.getLogger("autogen_core")
    agent_handler = AgentEventHandler(msg_logger, writer)
    core_logger.addHandler(agent_handler)
    # Send autogen_core verbose logs to file, not console
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{problem_id}_rerun.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    core_logger.addHandler(file_handler)
    core_logger.setLevel(logging.INFO)

    final_answer = None
    try:
        async for event in team.run_stream(task=history_messages):
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
                writer.add_step(agent=agent_name, content=event.content)
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
    tasks_by_id = {}
    for line in (data_dir / "validation.jsonl").read_text().splitlines():
        task = json.loads(line)
        tasks_by_id[task["task_id"]] = task

    file_to_task_id = {}
    for t in trajectories:
        file_to_task_id[t["instance_id"]] = t["filename"].replace(".json", "")

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
            problem_number=int(problem_id.split("_")[1]), agents=AGENT_DEFINITIONS,
        )
        live_log_dir = Path("logs") / "non_graph_reruns"
        writer.enable_live_log(live_log_dir / f"{task_id}.json")

        entry = {
            "problem_id": problem_id, "instance_id": instance_id,
            "critical_steps": result["critical_steps"],
            "last_critical_step": last_critical,
        }

        try:
            t0 = time.time()
            final_answer = await asyncio.wait_for(
                _rerun_single(task, history_messages, runner, writer, problem_id=problem_id), timeout=timeout,
            )
            latency = time.time() - t0
            writer.finalize(final_answer=final_answer)
            entry.update({
                "rerun_answer": final_answer,
                "latency_seconds": round(latency, 2),
                "status": "success",
            })
            log.info("  -> rerun answer: %s (%.1fs)", final_answer, latency)
        except asyncio.TimeoutError:
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
        lines.append(f"  What went wrong: {node['what_went_wrong']}")
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

def main():
    parser = argparse.ArgumentParser(description="Identify errors, fix, and rerun (no graph)")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], default=None)
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
        p1 = run_phase1(trajectories, args.api_url, args.model)
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
        s = p5["summary"]
        log.info("Flipped to correct: %d / %d", s["flipped_to_correct"], s["total_failures_analyzed"])
        log.info("Projected accuracy: %d / 10 (%.0f%%)",
                 s["projected_total_correct"], s["projected_total_accuracy"] * 100)


if __name__ == "__main__":
    main()
