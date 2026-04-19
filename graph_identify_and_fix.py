import numpy as np
"""Error propagation graph: identify errors, build causal error graph, fix ALL, rerun.

Improved pipeline:
- Phase 1: Identify ALL error nodes (LLM).
- Phase 2: Build ERROR PROPAGATION GRAPH via pairwise counterfactual LLM calls.
  Nodes = error steps only. Edges = "error at u caused error at v".
  Find root causes = error nodes with no incoming DEPENDENT edges.
  PARTIAL dependencies are also treated as root causes.
- Phase 3: Fix ALL error steps (with ground truth steering).
- Phase 4: Rerun from last error step.
- Phase 5: Score results.
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
DEFAULT_OUTPUT_DIR = "graph_error_analysis"
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
    "You are given the CORRECT answer for reference so you can precisely identify "
    "which steps caused the wrong answer. Identify ALL steps that contain errors — "
    "not just the first one. Return structured JSON only."
)

PHASE1_USER = """\
TASK: Identify ALL steps in this trajectory where an agent made an error that
contributed to producing the wrong final answer.

The agents were trying to answer this question:
{query}

The agents produced this (WRONG) answer: {final_answer}

THE CORRECT ANSWER IS: {ground_truth}

Use the correct answer as a reference to determine where the agents went off track.
A step is an error only if it contributed to the trajectory producing the wrong
answer instead of the correct one.

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
            ground_truth=traj["ground_truth"],
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
            seen_steps = set()
            deduped = []
            for n in valid:
                if n["step"] not in seen_steps:
                    seen_steps.add(n["step"])
                    deduped.append(n)
            entry["error_nodes"] = deduped
            entry["status"] = "success"
            log.info("  -> %d error nodes: %s", len(deduped), [n["step"] for n in deduped])

        analyses.append(entry)

    return {"analyses": analyses}


# ---------------------------------------------------------------------------
# Phase 2: Build error propagation graph + find root causes
# ---------------------------------------------------------------------------

PROPAGATION_SYSTEM = (
    "You are analyzing whether one error in a multi-agent trajectory "
    "caused another error. Answer with structured JSON only."
)

PROPAGATION_USER = """\
CONTEXT: A multi-agent team was trying to answer this question:
{query}

The trajectory PRODUCED THIS WRONG ANSWER: {wrong_final_answer}
THE CORRECT ANSWER IS: {ground_truth}

Here is the relevant portion of the trajectory between these two error steps:
{context_steps}

ERROR AT STEP {step_u} ({agent_u}, {type_u}):
{desc_u}

ERROR AT STEP {step_v} ({agent_v}, {type_v}):
{desc_v}

QUESTION: If step {step_u} had been executed CORRECTLY, would the error
at step {step_v} still have occurred?

Answer with a JSON object:
{{
  "relationship": "<INDEPENDENT|DEPENDENT|PARTIAL>",
  "reason": "<one sentence explanation>"
}}

- INDEPENDENT: The error at step {step_v} would still occur even if step {step_u}
  were correct. They are caused by different underlying issues.
- DEPENDENT: The error at step {step_v} was caused by or made inevitable by
  the error at step {step_u}. Fixing step {step_u} would likely prevent step {step_v}'s error.
- PARTIAL: The error at step {step_u} contributed to step {step_v}'s error, but
  step {step_v} also has its own independent issues.

Output only the JSON object."""


def find_root_causes(error_steps, dependent_edges, partial_edges):
    """Find root causes of the error propagation DAG.

    Root causes = error nodes with no incoming DEPENDENT edges.
    Nodes with only PARTIAL incoming edges are also root causes
    (they have their own independent issues).
    """
    has_full_parent = set()
    for u, v in dependent_edges:
        has_full_parent.add(v)

    root_causes = [n for n in error_steps if n not in has_full_parent]

    # Partial dependencies are ALSO root causes (they have independent issues)
    for u, v in partial_edges:
        if v not in root_causes:
            root_causes.append(v)

    return sorted(root_causes)


def build_error_clusters(error_steps, dependent_edges, partial_edges):
    """Group errors into clusters based on DEPENDENT edges.

    Each cluster is a connected component in the DEPENDENT-edge subgraph.
    The root cause of each cluster is the node with no incoming DEPENDENT edge.
    """
    if not error_steps:
        return []

    # Build adjacency for DEPENDENT edges only
    children = {}
    parents = {}
    for s in error_steps:
        children[s] = []
        parents[s] = []
    for u, v in dependent_edges:
        if u in children and v in parents:
            children[u].append(v)
            parents[v].append(u)

    # Find connected components via BFS on undirected DEPENDENT edges
    visited = set()
    clusters = []
    for step in sorted(error_steps):
        if step in visited:
            continue
        # BFS
        component = []
        queue = [step]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for child in children.get(node, []):
                if child not in visited:
                    queue.append(child)
            for parent in parents.get(node, []):
                if parent not in visited:
                    queue.append(parent)

        component.sort()
        # Root of this cluster = nodes with no DEPENDENT parent within the cluster
        component_set = set(component)
        roots = [s for s in component if not any(p in component_set for p in parents.get(s, []))]
        root = roots[0] if roots else component[0]

        clusters.append({
            "cluster_id": len(clusters) + 1,
            "root_cause_step": root,
            "error_steps": component,
        })

    return clusters


def run_phase2(phase1_data, trajectories, api_url, model):
    """Build error propagation graph via pairwise counterfactual LLM calls."""
    traj_by_id = {t["instance_id"]: t for t in trajectories}
    results = []

    for analysis in phase1_data["analyses"]:
        problem_id = analysis["problem_id"]
        instance_id = analysis["instance_id"]
        error_nodes = analysis.get("error_nodes", [])
        error_steps = [n["step"] for n in error_nodes]

        if not error_nodes:
            results.append({
                "problem_id": problem_id,
                "instance_id": instance_id,
                "filename": analysis["filename"],
                "all_error_steps": [],
                "error_graph": {"nodes": [], "edges": []},
                "clusters": [],
                "critical_nodes": [],
                "critical_steps": [],
                "last_error_step": None,
                "status": "no_errors",
            })
            log.info("%s: 0 errors -> no graph", problem_id)
            continue

        # Single error — trivially its own root cause
        if len(error_nodes) == 1:
            results.append({
                "problem_id": problem_id,
                "instance_id": instance_id,
                "filename": analysis["filename"],
                "all_error_steps": error_steps,
                "error_graph": {
                    "nodes": error_steps,
                    "edges": [],
                },
                "clusters": [{
                    "cluster_id": 1,
                    "root_cause_step": error_steps[0],
                    "error_steps": error_steps,
                }],
                "critical_nodes": [error_nodes[0]],
                "critical_steps": error_steps,
                "last_error_step": error_steps[0],
                "status": "success",
            })
            log.info("%s: 1 error -> 1 root cause: %s", problem_id, error_steps)
            continue

        traj = traj_by_id[instance_id]
        error_by_step = {n["step"]: n for n in error_nodes}

        # Build pairwise error propagation graph
        edges = []  # {"source", "target", "relationship", "reason"}
        num_pairs = len(error_nodes) * (len(error_nodes) - 1) // 2
        log.info("Phase 2: %s — %d errors, %d pairwise comparisons",
                 problem_id, len(error_nodes), num_pairs)

        for i, node_u in enumerate(error_nodes):
            for node_v in error_nodes[i + 1:]:
                step_u = node_u["step"]
                step_v = node_v["step"]

                # Build context: steps between u and v (inclusive)
                context_steps_list = [s for s in traj["steps"]
                                      if step_u <= s["step"] <= step_v]
                context_text = format_trajectory(context_steps_list, truncate=False)

                user_prompt = PROPAGATION_USER.format(
                    query=analysis["query"],
                    wrong_final_answer=analysis.get("original_answer") or "(no answer produced)",
                    ground_truth=analysis.get("ground_truth", ""),
                    context_steps=context_text,
                    step_u=step_u, agent_u=node_u["agent"], type_u=node_u["error_type"],
                    desc_u=node_u["what_went_wrong"],
                    step_v=step_v, agent_v=node_v["agent"], type_v=node_v["error_type"],
                    desc_v=node_v["what_went_wrong"],
                )

                parsed, raw, usage = call_llm_with_retries(
                    api_url, model, PROPAGATION_SYSTEM, user_prompt,
                    f"{problem_id}_pair_{step_u}_{step_v}",
                )

                if parsed and "relationship" in parsed:
                    rel = parsed["relationship"].upper()
                    reason = parsed.get("reason", "")
                else:
                    rel = "INDEPENDENT"
                    reason = "LLM call failed, defaulting to independent"

                if rel in ("DEPENDENT", "PARTIAL"):
                    edges.append({
                        "source": step_u,
                        "target": step_v,
                        "relationship": rel,
                        "reason": reason,
                    })
                    log.info("  %d -> %d: %s (%s)", step_u, step_v, rel, reason[:60])

        # Separate edges by type
        dependent_edges = [(e["source"], e["target"]) for e in edges if e["relationship"] == "DEPENDENT"]
        partial_edges = [(e["source"], e["target"]) for e in edges if e["relationship"] == "PARTIAL"]

        # Find root causes
        root_causes = find_root_causes(error_steps, dependent_edges, partial_edges)

        # Build clusters
        clusters = build_error_clusters(error_steps, dependent_edges, partial_edges)

        critical_details = [error_by_step[s] for s in root_causes if s in error_by_step]
        last_error = max(error_steps)

        results.append({
            "problem_id": problem_id,
            "instance_id": instance_id,
            "filename": analysis["filename"],
            "all_error_steps": error_steps,
            "error_graph": {
                "nodes": error_steps,
                "edges": edges,
            },
            "clusters": clusters,
            "critical_nodes": critical_details,
            "critical_steps": root_causes,
            "last_error_step": last_error,
            "status": "success",
        })
        log.info("%s: %d errors, %d edges (%d dep, %d partial) -> %d root causes: %s, %d clusters",
                 problem_id, len(error_steps), len(edges), len(dependent_edges),
                 len(partial_edges), len(root_causes), root_causes, len(clusters))

    return {"results": results}


# ---------------------------------------------------------------------------
# Phase 3: Generate fixes for ALL error steps (with ground truth)
# ---------------------------------------------------------------------------

FIX_SYSTEM = "You are {agent_name}. {agent_system_prompt}"

FIX_USER = """\
CONVERSATION SO FAR:
{prior_context}

YOUR PREVIOUS RESPONSE AT THIS STEP WAS:
{original_content}

THIS RESPONSE WAS WRONG BECAUSE:
{what_went_wrong}

THE WRONG FINAL ANSWER THE TRAJECTORY PRODUCED: {wrong_final_answer}
THE CORRECT FINAL ANSWER IS: {ground_truth}

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


def run_phase3(trajectories, phase1_data, phase2_data, api_url, model):
    """Generate fixes for ALL error steps, not just critical nodes."""
    traj_by_id = {t["instance_id"]: t for t in trajectories}
    analysis_by_id = {a["instance_id"]: a for a in phase1_data["analyses"]}
    fixes = []

    for result in phase2_data["results"]:
        if result.get("status") != "success" or not result.get("all_error_steps"):
            fixes.append({
                "problem_id": result["problem_id"],
                "instance_id": result["instance_id"],
                "fixes": [], "status": "skipped",
            })
            continue

        problem_id = result["problem_id"]
        instance_id = result["instance_id"]
        traj = traj_by_id[instance_id]
        analysis = analysis_by_id[instance_id]
        steps = traj["steps"]
        ground_truth = traj["ground_truth"]

        error_by_step = {n["step"]: n for n in analysis.get("error_nodes", [])}
        # Only fix critical nodes (root causes from error propagation graph)
        critical_steps = sorted(result.get("critical_steps", []))

        applied_fixes = {}
        node_fixes = []

        for step_idx in critical_steps:
            node = error_by_step.get(step_idx)
            if node is None:
                continue

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
                wrong_final_answer=traj.get("final_answer") or "(no answer produced)",
                ground_truth=ground_truth,
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
# Phase 4: Rerun with Magentic-One (resume from last error step)
# ---------------------------------------------------------------------------

def _build_history_messages(steps, all_fixes, last_error_step):
    """Build history with ALL fixes applied, up to the last error step."""
    from autogen_agentchat.messages import TextMessage as AutogenTextMessage
    fix_map = {f["step"]: f["corrected_content"] for f in all_fixes if f.get("status") == "success"}
    messages = []
    for s in steps:
        if s["step"] > last_error_step:
            break
        content = fix_map.get(s["step"], s["content"])
        messages.append(AutogenTextMessage(content=content, source=s["agent"]))
    return messages


async def _rerun_single(task, history_messages, runner, writer, problem_id="unknown"):
    from runners.magentic_one_runner import MessageLogger, AgentEventHandler, _parse_args
    from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
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
    msg_logger = MessageLogger(f"{problem_id}_graph_rerun")

    web_surfer = MultimodalWebSurfer("WebSurfer", model_client=runner.client, headless=True)
    file_surfer = FileSurfer("FileSurfer", model_client=runner.client)
    coder = AssistantAgent(
        "Coder",
        model_client=runner.client,
        system_message=(
            "You are a helpful AI assistant.\n"
            "Solve tasks using your coding and language skills.\n"
            "When you need to write code, you MUST use a markdown code block with the language tag.\n"
            "For Python, use:\n```python\nprint('hello')\n```\n"
            "For shell, use:\n```sh\necho hello\n```\n"
            "NEVER use tool call syntax like call:executor or <|tool_call>. "
            "ALWAYS wrap code in ```python or ```sh blocks. "
            "The Executor agent will run your code blocks automatically.\n"
            "Don't include multiple code blocks in one response. "
            "Use 'print' for output. Check execution results and fix errors if needed."
        ),
        description=(
            "A helpful and general-purpose AI assistant that has strong language skills, "
            "Python skills, and Linux command line skills."
        ),
    )
    work_dir = Path(tempfile.mkdtemp(prefix=f"grerun_{task_id[:8]}_"))
    code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
    executor = CodeExecutorAgent("Executor", code_executor=code_executor)

    team = MagenticOneGroupChat(
        participants=[web_surfer, file_surfer, coder, executor],
        model_client=runner.client, max_turns=DEFAULT_MAX_TURNS,
    )

    core_logger = logging.getLogger("autogen_core")
    agent_handler = AgentEventHandler(msg_logger, writer)
    core_logger.addHandler(agent_handler)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{problem_id}_graph_rerun.log", mode="w")
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

        if result.get("status") != "success" or not result.get("all_error_steps"):
            simulations.append({
                "problem_id": problem_id, "instance_id": instance_id,
                "status": "skipped", "reason": "no errors found",
            })
            continue

        if not result.get("critical_steps"):
            simulations.append({
                "problem_id": problem_id, "instance_id": instance_id,
                "status": "skipped", "reason": "no root nodes",
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

        # Resume from last critical step (root cause), not last error step
        last_critical = max(result.get("critical_steps", [])) if result.get("critical_steps") else result["last_error_step"]
        all_fixes = fix_data["fixes"]

        history_messages = _build_history_messages(traj["steps"], all_fixes, last_critical)
        num_fixes = sum(1 for f in all_fixes if f.get("status") == "success")
        log.info("Phase 4: %s — %d history messages (up to step %d, %d fixes applied)",
                 problem_id, len(history_messages), last_critical, num_fixes)

        writer = TrajectoryWriter(
            task=task, framework="magentic-one-graph-rerun", model=model,
            output_dir=str(rerun_dir), instance_id=instance_id + "_graph_rerun",
            problem_number=orig_num, agents=AGENT_DEFINITIONS,
        )
        writer.enable_live_log()

        entry = {
            "problem_id": problem_id, "instance_id": instance_id,
            "all_error_steps": result["all_error_steps"],
            "critical_steps": result["critical_steps"],
            "last_critical_step": last_critical,
            "num_fixes_applied": num_fixes,
            "num_clusters": len(result.get("clusters", [])),
        }

        try:
            t0 = time.time()
            rerun_task = asyncio.ensure_future(
                _rerun_single(task, history_messages, runner, writer, problem_id=problem_id),
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
            "error_graph_edges": critical.get("error_graph", {}).get("edges", []),
            "clusters": critical.get("clusters", []),
            "critical_steps": critical.get("critical_steps", []),
            "last_error_step": critical.get("last_error_step"),
            "num_fixes_applied": sim.get("num_fixes_applied", 0),
        }
        results.append(result)
        write_report(result, analysis, critical, fix_data, sim, output_dir)

    original_correct_total = sum(1 for r in results if r["original_correct"])
    summary = {
        "total_failures_analyzed": total,
        "flipped_to_correct": flipped,
        "remained_wrong": total - flipped,
        "original_correct_total": original_correct_total,
        "projected_total_correct": original_correct_total + flipped,
        "projected_total_accuracy": round(
            (original_correct_total + flipped) / max(total + original_correct_total, 1), 2
        ),
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
        f"ALL ERROR STEPS:   {result['all_error_steps']}",
        f"ROOT CAUSES:       {result['critical_steps']}",
        f"LAST ERROR STEP:   {result.get('last_error_step', 'N/A')}",
        f"FIXES APPLIED:     {result.get('num_fixes_applied', 0)}",
        "",
    ]

    # Show error propagation graph edges
    edges = result.get("error_graph_edges", [])
    if edges:
        lines.append("ERROR PROPAGATION GRAPH:")
        for e in edges:
            lines.append(f"  {e['source']} -> {e['target']} [{e['relationship']}]: {e.get('reason', '')[:100]}")
        lines.append("")

    # Show clusters
    for cluster in result.get("clusters", []):
        lines.append(f"--- Cluster {cluster['cluster_id']} (root: step {cluster['root_cause_step']}) ---")
        lines.append(f"  Steps: {cluster['error_steps']}")
        lines.append("")

    # Show fixes
    lines.append("FIXES:")
    for fix in fix_data.get("fixes", []):
        step = fix.get("step", "?")
        etype = fix.get("error_type", "?")
        status = fix.get("status", "?")
        wrong = fix.get("what_went_wrong", "")
        lines.append(f"  Step {step} ({etype}) [{status}]: {wrong[:150]}")
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
    parser = argparse.ArgumentParser(
        description="Error propagation graph: identify errors, build causal graph, fix all, rerun"
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], default=None)
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
    p2_path = output_dir / "step2_error_graph.json"
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
        log.info("=== PHASE 2: Build error propagation graph ===")
        p2 = run_phase2(p1, trajectories, args.api_url, args.model)
        p2_path.write_text(json.dumps(p2, indent=2, ensure_ascii=False))

    if args.phase is None or args.phase == 3:
        p1 = json.loads(p1_path.read_text())
        p2 = json.loads(p2_path.read_text())
        log.info("=== PHASE 3: Generate fixes for ALL error steps ===")
        p3 = run_phase3(trajectories, p1, p2, args.api_url, args.model)
        p3_path.write_text(json.dumps(p3, indent=2, ensure_ascii=False))

    if args.phase is None or args.phase == 4:
        p2 = json.loads(p2_path.read_text())
        p3 = json.loads(p3_path.read_text())
        log.info("=== PHASE 4: Rerun with Magentic-One (from last error step) ===")
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
        log.info("Projected accuracy: %d / %d (%.0f%%)",
                 s["projected_total_correct"],
                 s["total_failures_analyzed"] + s["original_correct_total"],
                 s["projected_total_accuracy"] * 100)


if __name__ == "__main__":
    main()
