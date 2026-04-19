"""Counterfactual rerun for OWL (CAMEL ChatAgent) with the same percentile
k-sweep logic as parsimony_experiment.py.

OWL is a single ChatAgent with linear memory, so the rerun is straightforward:
  1. Build a fresh ChatAgent with the same system prompt + tools.
  2. Pre-populate its memory with trajectory steps [0..last_fixed_step],
     substituting corrected content for fixed steps.
  3. Call agent.step(...) on a continuation prompt to drive the agent forward.

Protocol per trajectory (mirrors parsimony_experiment):
  1. Rank error nodes by step ascending; drop the terminal step.
  2. Sweep k in {1, ceil(n*0.33), ceil(n*0.66), n} (ascending, unique).
  3. For each k: generate cumulative fixes (parsimony_experiment.generate_fixes),
     rerun with memory-restoration, check flip.
  4. First flip wins; exhaustion = unsalvageable.
  5. Confirmation rerun at k*.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import SearchToolkit
from camel.types import ModelPlatformType, OpenAIBackendRole

from linear_identify_and_fix import (
    DEFAULT_API_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT, MAX_TOKENS,
    FIX_SYSTEM, call_llm_raw, format_step_content, format_trajectory,
    load_failed_trajectories,
)
from parsimony_experiment import rank_error_steps


MAX_K = 100  # cap k-sweep to avoid pathological long trajectories (e.g. 0008)


def compute_k_levels(n: int) -> list:
    """OWL override: sweep only endpoints — minimal fix (k=1) and full fix.

    The "full fix" is capped at MAX_K so trajectories with hundreds of error
    nodes (e.g. gaia_validation_0008 with 580 tool-step candidates) don't
    balloon into 500+ agent reruns."""
    if n <= 0:
        return []
    k_full = min(n, MAX_K)
    if k_full == 1:
        return [1]
    return [1, k_full]
from runners.owl_runner import OWL_AGENT_DEFINITIONS
from trajectory.evaluator import evaluate_answer
from trajectory.writer import TrajectoryWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("owl_cf")

MAX_FOLLOWUP_TURNS = 8

OWL_SYSTEM_MSG = (
    "You are a helpful AI assistant solving benchmark tasks. "
    "You have access to web search tools to find information online. "
    "Use search_tavily, search_duckduckgo, or search_wiki to look up facts you need. "
    "Think step by step. When you have the final answer, state it clearly as: "
    "FINAL ANSWER: <answer>\n"
    "The answer should be concise - just the value, name, number, or short phrase requested."
)

_FINAL_RE = re.compile(r"FINAL ANSWER[:\s]+(.+?)(?:\n|$)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# OWL-specific fix prompts — we fix ONLY tool steps:
#   * TOOL CALL  (Assistant step with a tool_calls entry)
#   * TOOL RESULT (ComputerTerminal step produced by the search tools)
# Reasoning-step fixes tend to rephrase without redirecting, so we skip them.
# ---------------------------------------------------------------------------

FIX_USER_OWL_TOOL_CALL = """\
CONVERSATION SO FAR:
{prior_context}

YOUR PREVIOUS TOOL CALL AT THIS STEP WAS:
{original_content}

THIS STEP WAS FLAGGED AS A {error_type} ERROR.

THE WRONG FINAL ANSWER THE TRAJECTORY PRODUCED: {wrong_final_answer}

Produce a corrected TOOL CALL. You are acting AS the assistant at this step,
but without knowledge of the ground-truth answer. Your goal is to fix the
retrieval decision so that the downstream agent has a better chance of
reaching the correct answer.

CRITICAL RULES:
- Output EXACTLY ONE tool call, formatted as: tool_name({{"query": "..."}}).
- Valid tools: search_tavily, search_duckduckgo, search_wiki.
- You may change the tool, the query, or both. Pick the combination most
  likely to return helpful, on-point evidence for the user's question.
- Do NOT leak the final answer. You do not know it.
- Do NOT include information you could not have known at this point in the
  conversation. Use only what is in the conversation so far.
- Output ONLY the tool call string, no preamble, no explanation."""


FIX_USER_OWL_TOOL_RESULT = """\
CONVERSATION SO FAR:
{prior_context}

THE ORIGINAL TOOL CALL AT THE PREVIOUS STEP WAS:
{tool_call_preceding}

THE TOOL RETURNED (THIS STEP, FLAGGED AS A {error_type} ERROR):
{original_content}

THE WRONG FINAL ANSWER THE TRAJECTORY PRODUCED: {wrong_final_answer}

Produce a REALISTIC alternative result that this tool (same tool, same
arguments) MIGHT have returned — e.g. a different page from the same query,
a snippet that is more relevant to the user's question.

CRITICAL RULES:
- This is a counterfactual observation, not ground truth. You are NOT
  allowed to invent content that directly states the final answer. Inject
  evidence that plausibly exists in real search results and that would help
  the downstream agent reach the answer through its own reasoning.
- Match the shape and tone of a genuine search-tool result (list of
  snippets, short factual text, etc.). Do not editorialize or reason.
- Do NOT include the phrase "FINAL ANSWER" or restate the user's question.
- Match the length of the original result. If the original was
  {original_length} characters, stay similar; do not expand more than 2x.
Output ONLY the alternative tool result text, no preamble."""


def _preceding_tool_call(steps: list, result_step_idx: int) -> str:
    """Find the Assistant tool_calls step immediately before `result_step_idx`."""
    for s in reversed(steps):
        if s["step"] >= result_step_idx:
            continue
        if s.get("agent") == "Assistant" and s.get("tool_calls"):
            return s.get("content", "")
        # Stop if we walk back past another tool result without finding a call
        if s.get("agent") == "ComputerTerminal":
            break
    return ""


def _is_tool_call_step(step: dict) -> bool:
    return step.get("agent") == "Assistant" and bool(step.get("tool_calls"))


def _is_tool_result_step(step: dict) -> bool:
    return step.get("agent") == "ComputerTerminal"


def generate_fixes_owl(traj: dict, nodes_by_step_asc: list,
                        api_url: str, model: str,
                        api_key: str | None = None) -> list:
    """Cumulative fix generation for OWL, constrained to tool steps.

    For each candidate node:
      - if the step is a tool call: use FIX_USER_OWL_TOOL_CALL
      - if the step is a tool result: use FIX_USER_OWL_TOOL_RESULT
      - otherwise: skip (with a log warning) — caller should have filtered
    Earlier fixes in the sequence feed into later `prior_context`.
    """
    steps = traj["steps"]
    applied: dict[int, str] = {}
    out = []
    for node in nodes_by_step_asc:
        step_idx = node["step"]
        agent = node["agent"]
        step_data = next(s for s in steps if s["step"] == step_idx)

        prior = []
        for s in steps:
            if s["step"] >= step_idx:
                break
            ns = dict(s)
            if ns["step"] in applied:
                ns["content"] = applied[ns["step"]]
            prior.append(ns)

        prior_ctx = format_trajectory(prior)
        original = format_step_content(step_data["content"])

        sys_prompt = FIX_SYSTEM.format(
            agent_name=agent,
            agent_system_prompt=traj["agent_prompts"].get(agent, ""),
        )

        if _is_tool_call_step(step_data):
            step_kind = "tool_call"
            usr_prompt = FIX_USER_OWL_TOOL_CALL.format(
                prior_context=prior_ctx,
                original_content=original,
                error_type=node.get("error_type", "UNKNOWN"),
                wrong_final_answer=traj.get("final_answer") or "(no answer produced)",
            )
        elif _is_tool_result_step(step_data):
            step_kind = "tool_result"
            usr_prompt = FIX_USER_OWL_TOOL_RESULT.format(
                prior_context=prior_ctx,
                tool_call_preceding=_preceding_tool_call(steps, step_idx),
                original_content=original,
                error_type=node.get("error_type", "UNKNOWN"),
                wrong_final_answer=traj.get("final_answer") or "(no answer produced)",
                original_length=len(step_data["content"]),
            )
        else:
            log.warning("skip non-tool step %d (%s) from fix generation",
                        step_idx, agent)
            continue

        fix_tokens = min(MAX_TOKENS, max(512, len(step_data["content"]) * 2))
        raw, _ = call_llm_raw(api_url, model, sys_prompt, usr_prompt,
                               max_tokens=fix_tokens, api_key=api_key)
        corrected = raw.strip()
        applied[step_idx] = corrected
        out.append({
            "step": step_idx,
            "agent": agent,
            "error_type": node.get("error_type"),
            "step_kind": step_kind,
            "original_content": step_data["content"],
            "corrected_content": corrected,
        })
    return out


def _has_final(text: str) -> bool:
    return bool(_FINAL_RE.search(text or ""))


def _extract_answer(text: str) -> str | None:
    if not text:
        return None
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1][:200] if lines else None


def _build_agent(model_platform, model_name: str, api_url: str):
    model_backend = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_name,
        url=api_url.replace("/chat/completions", "").rsplit("/v1", 1)[0] + "/v1",
        api_key="unused",
        model_config_dict={"max_tokens": 4096, "temperature": 0},
    )
    toolkit = SearchToolkit()
    tools = [t for t in toolkit.get_tools()
             if t.get_function_name() in ("search_tavily", "search_duckduckgo", "search_wiki")]
    agent = ChatAgent(system_message=OWL_SYSTEM_MSG, model=model_backend, tools=tools)
    # Expose a name->callable map on the agent so replay can live-execute fixed
    # tool calls instead of relying on the stored (now-stale) observation.
    agent._tip_tools = {t.get_function_name(): t.func for t in tools}
    return agent


_FIX_TOOL_CALL_RE = re.compile(r"^\s*(\w+)\s*\((\{.*\})\s*\)\s*$", re.DOTALL)


def _parse_fixed_tool_call(content: str):
    """Parse 'tool_name({"arg": ...})' form OR a raw JSON dict {"tool_name":
    "...", "tool_args": {...}} OR a bare JSON args dict (tool_name inferred
    from prior stored step). Returns (name, args) or None."""
    if not content:
        return None
    m = _FIX_TOOL_CALL_RE.match(content.strip())
    if m:
        name, args_blob = m.group(1), m.group(2)
        try:
            return name, json.loads(args_blob)
        except Exception:
            return None
    # bare JSON — caller supplies default name
    try:
        obj = json.loads(content.strip())
        if isinstance(obj, dict) and "tool_name" in obj:
            return obj["tool_name"], obj.get("tool_args") or {}
    except Exception:
        pass
    return None


def _execute_fixed_tool(agent: ChatAgent, name: str, args: dict) -> str:
    """Look up `name` in agent._tip_tools and call it with `args`. Returns a
    string rendering of the result (or an error message)."""
    tools_map = getattr(agent, "_tip_tools", {}) or {}
    func = tools_map.get(name)
    if func is None:
        return f"[error] unknown tool '{name}' — available: {sorted(tools_map)}"
    try:
        result = func(**(args or {}))
    except Exception as exc:
        return f"[error] tool {name} raised {type(exc).__name__}: {exc}"
    # Stringify whatever the tool returns so it lands cleanly in memory.
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def _prepopulate_memory(agent: ChatAgent, traj: dict, fixes: list,
                         last_fixed_step: int) -> None:
    """Write trajectory entries [0..last_fixed_step] into agent memory,
    substituting corrected content for fixed steps.

    When a fix replaces an Assistant tool_call, we EXECUTE the fixed call
    live via the agent's toolkit and use the resulting observation in memory
    (instead of the stored, now-stale observation). This is what makes
    tool-swap and tool-args fixes meaningful — the downstream agent sees a
    real observation, not a fabricated one.
    """
    fix_map = {f["step"]: f["corrected_content"] for f in fixes}
    steps = traj["steps"]
    idx_to_step = {s["step"]: s for s in steps}

    # Track fixed tool_calls so the immediately-following ComputerTerminal
    # step knows to use a live execution result.
    pending_live_result: dict[int, str] = {}

    for s in steps:
        idx = s["step"]
        if idx > last_fixed_step:
            break
        content = fix_map.get(idx, s["content"])
        agent_name = s.get("agent", "")

        if agent_name == "user":
            msg = BaseMessage.make_user_message(role_name="user", content=content)
            agent.update_memory(msg, OpenAIBackendRole.USER)

        elif agent_name in ("Assistant", "assistant"):
            msg = BaseMessage.make_assistant_message(
                role_name="Assistant", content=content)
            agent.update_memory(msg, OpenAIBackendRole.ASSISTANT)
            # If this Assistant step was FIXED AND parses as a tool_call,
            # execute it now and stash the live result for the next tool
            # step.
            if idx in fix_map:
                parsed = _parse_fixed_tool_call(content)
                # Fall back: the fix is bare JSON args ({"entity": "..."} or
                # {"query": "..."}). Infer the target tool from arg shape:
                #   - "entity" is search_wiki-only
                #   - otherwise reuse the stored step's original tool name
                if parsed is None:
                    try:
                        args_only = json.loads(content)
                        if isinstance(args_only, dict):
                            orig_tool = None
                            if s.get("tool_calls"):
                                orig_tool = (s["tool_calls"][0] or {}).get("tool")
                            if "entity" in args_only and "query" not in args_only:
                                inferred = "search_wiki"
                            elif orig_tool:
                                inferred = orig_tool
                            else:
                                inferred = None
                            if inferred:
                                parsed = (inferred, args_only)
                    except Exception:
                        parsed = None
                if parsed is not None:
                    name, args = parsed
                    log.info("[replay] executing fixed tool call at step %d: %s(%s)",
                             idx, name, args)
                    live = _execute_fixed_tool(agent, name, args)
                    pending_live_result[idx] = live

        elif agent_name == "ComputerTerminal":
            tool_name = "unknown_tool"
            tr_list = s.get("tool_results") or []
            if tr_list:
                tool_name = tr_list[0].get("tool") or tool_name
            # Use live result if the preceding assistant step was a fixed
            # tool_call; otherwise use stored / fixed content as before.
            live = None
            for prev_idx in range(idx - 1, -1, -1):
                prev = idx_to_step.get(prev_idx)
                if prev is None or prev.get("agent", "") not in ("Assistant", "assistant"):
                    continue
                if prev_idx in pending_live_result:
                    live = pending_live_result.pop(prev_idx)
                break
            observed = live if live is not None else content
            annotated = f"[observation from {tool_name}]\n{observed}"
            msg = BaseMessage.make_assistant_message(
                role_name="Assistant", content=annotated)
            agent.update_memory(msg, OpenAIBackendRole.ASSISTANT)
        # Other step types (Planner-thought, llm_call marker) are ignored.


def _continuation_prompt(traj: dict, last_fixed_step: int) -> str:
    """Find the next user message after `last_fixed_step` in the original
    trajectory; fall back to a generic "continue" if none."""
    for s in traj["steps"]:
        if s["step"] > last_fixed_step and s.get("agent") == "user":
            return s["content"]
    return ("Please continue your analysis. Use search tools if you need more "
            "information. When done, state: FINAL ANSWER: <answer>")


async def rerun_owl(traj: dict, fixes: list, task: dict, writer: TrajectoryWriter,
                     model_name: str, api_url: str, timeout: int) -> dict:
    """Run OWL with fixes applied by memory restoration.

    Returns dict with final_answer / status / latency.
    """
    last_fixed = max(f["step"] for f in fixes)
    t0 = time.time()

    agent = _build_agent(
        ModelPlatformType.OPENAI_COMPATIBLE_MODEL, model_name, api_url)

    writer.add_step(agent="system",
                    content=f"[replay] pre-populating memory up to step {last_fixed} "
                             f"with {len(fixes)} fix(es)")

    _prepopulate_memory(agent, traj, fixes, last_fixed)

    user_prompt = _continuation_prompt(traj, last_fixed)
    writer.add_step(agent="user", content=user_prompt)
    user_msg = BaseMessage.make_user_message(role_name="user", content=user_prompt)

    try:
        # Run in a thread so we can apply a timeout
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: agent.step(user_msg)),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return {"final_answer": None, "status": "timeout",
                "latency_seconds": round(time.time() - t0, 2),
                "last_fixed_step": last_fixed}
    except Exception as e:
        log.warning("agent.step failed: %s", e)
        return {"final_answer": None, "status": "failed", "error": str(e),
                "latency_seconds": round(time.time() - t0, 2),
                "last_fixed_step": last_fixed}

    content = ""
    if response and response.msgs:
        content = response.msgs[0].content
        writer.add_step(agent="Assistant", content=content[:2000])

    turns = 0
    while turns < MAX_FOLLOWUP_TURNS and not _has_final(content):
        turns += 1
        follow_up_text = (
            "Please continue your analysis. Use search tools if you need more "
            "information. When done, state: FINAL ANSWER: <answer>")
        writer.add_step(agent="user", content=follow_up_text)
        fu = BaseMessage.make_user_message(role_name="user", content=follow_up_text)
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: agent.step(fu)),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return {"final_answer": _extract_answer(content), "status": "timeout",
                    "latency_seconds": round(time.time() - t0, 2),
                    "last_fixed_step": last_fixed}
        except Exception as e:
            log.warning("agent.step (follow-up) failed: %s", e)
            return {"final_answer": _extract_answer(content), "status": "failed",
                    "error": str(e),
                    "latency_seconds": round(time.time() - t0, 2),
                    "last_fixed_step": last_fixed}
        if response and response.msgs:
            content = response.msgs[0].content
            writer.add_step(agent="Assistant", content=content[:2000])

    return {"final_answer": _extract_answer(content), "status": "success",
            "latency_seconds": round(time.time() - t0, 2),
            "last_fixed_step": last_fixed}


async def run_trajectory(traj: dict, error_nodes: list, task: dict,
                          output_dir: Path, problem_number: int,
                          api_url: str, model: str, timeout: int,
                          fix_api_url: str, fix_model: str,
                          fix_api_key: str | None) -> dict:
    iid = traj["instance_id"]
    log.info("=== %s (problem_%02d) ===", iid, problem_number)
    log.info("  num_steps=%d n_errors=%d", len(traj["steps"]), len(error_nodes))

    gt = traj["ground_truth"]
    wrong_answer = traj.get("final_answer") or ""

    # Restrict candidates to tool steps only (tool call or tool result).
    # Reasoning-step fixes rarely redirect the trajectory; tool steps are
    # where information actually enters the conversation.
    step_by_idx = {s["step"]: s for s in traj["steps"]}
    tool_error_nodes = [
        n for n in error_nodes
        if (s := step_by_idx.get(n["step"])) is not None
        and (_is_tool_call_step(s) or _is_tool_result_step(s))
    ]
    log.info("  tool-only filter: %d/%d error nodes kept",
             len(tool_error_nodes), len(error_nodes))
    ranked = rank_error_steps(traj, tool_error_nodes)
    ordered_nodes = [r["node"] for r in ranked]
    n = len(ordered_nodes)
    levels = compute_k_levels(n)
    log.info("  ordering (earliest-first, excl. terminal): %s",
             [r["step"] for r in ranked])
    log.info("  k-levels: %s", levels)

    result = {
        "instance_id": iid, "problem_number": problem_number,
        "num_steps": len(traj["steps"]), "ground_truth": gt,
        "wrong_answer": wrong_answer,
        "ordering": [{"step": r["step"]} for r in ranked],
        "k_levels_tested": levels, "levels": [],
        "k_star": None, "flip_verified": False, "confirmation_flipped": None,
        "minimal_critical_set": [], "fixes_at_kstar": [],
    }
    if n == 0:
        log.info("  => SKIPPED (0 candidate error nodes after filtering)")
        result["status"] = "skipped_no_candidates"
        return result

    async def _test_level(k: int):
        selected = sorted(ordered_nodes[:k], key=lambda x: x["step"])
        log.info("  -> level k=%d steps=%s", k, [s["step"] for s in selected])
        try:
            fixes = generate_fixes_owl(traj, selected, fix_api_url, fix_model,
                                         api_key=fix_api_key)
        except Exception as e:
            log.warning("    fix generation failed: %s", e)
            return False, None, {"k": k, "selected_steps": [s["step"] for s in selected],
                                  "status": "fix_failed", "error": str(e)}

        rerun_dir = output_dir / f"reruns_k{k}"
        rerun_dir.mkdir(parents=True, exist_ok=True)
        writer = TrajectoryWriter(
            task=task, framework="owl-counterfactual", model=model,
            output_dir=str(rerun_dir), instance_id=iid + f"_k{k}",
            problem_number=problem_number, agents=OWL_AGENT_DEFINITIONS,
        )
        writer.enable_live_log()
        rerun = await rerun_owl(traj, fixes, task, writer, model, api_url, timeout)
        writer.finalize(final_answer=rerun.get("final_answer"),
                        error=rerun.get("error"))
        flipped = (rerun.get("status") == "success"
                   and rerun.get("final_answer")
                   and evaluate_answer(rerun.get("final_answer"), gt))
        rec = {
            "k": k, "selected_steps": [s["step"] for s in selected],
            "rerun_answer": rerun.get("final_answer"),
            "rerun_status": rerun.get("status"),
            "flipped": bool(flipped),
            "last_fixed_step": rerun.get("last_fixed_step"),
            "latency_seconds": rerun.get("latency_seconds"),
            "fixes": fixes,
        }
        log.info("    answer=%s flipped=%s", rerun.get("final_answer"), flipped)
        return bool(flipped), fixes, rec

    best_k, best_fixes = None, None
    for k in levels:
        flipped, fixes, rec = await _test_level(k)
        result["levels"].append(rec)
        if flipped:
            best_k, best_fixes = k, fixes
            break

    if best_k is None:
        log.info("  => UNSALVAGEABLE across %s", levels)
        result["status"] = "unsalvageable"
        return result

    result["k_star"] = best_k
    result["flip_verified"] = True
    result["minimal_critical_set"] = sorted(s["step"] for s in ordered_nodes[:best_k])
    result["fixes_at_kstar"] = best_fixes
    result["status"] = "success"

    # Confirmation rerun
    log.info("  -> confirmation rerun at k*=%d", best_k)
    conf_dir = output_dir / "reruns_confirm"
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_writer = TrajectoryWriter(
        task=task, framework="owl-counterfactual", model=model,
        output_dir=str(conf_dir), instance_id=iid + f"_confirm_k{best_k}",
        problem_number=problem_number, agents=OWL_AGENT_DEFINITIONS,
    )
    conf_writer.enable_live_log()
    # Confirmation rerun uses the SAME agent client as the replay — the fix
    # LLM is separate. Reuse the replay api_url/model for the agent.
    conf = await rerun_owl(traj, best_fixes, task, conf_writer, model, api_url, timeout)
    conf_writer.finalize(final_answer=conf.get("final_answer"))
    conf_flipped = (conf.get("status") == "success" and conf.get("final_answer")
                    and evaluate_answer(conf.get("final_answer"), gt))
    result["confirmation_flipped"] = bool(conf_flipped)
    result["confirmation_answer"] = conf.get("final_answer")
    log.info("  => FLIPPED at k*=%d confirmation=%s", best_k, conf_flipped)
    return result


async def main_async(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "owl_cf.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(fh)

    step1 = json.loads(Path(args.phase1_file).read_text())
    analyses_by_iid = {a["instance_id"]: a for a in step1.get("analyses", [])}

    trajectories = load_failed_trajectories(input_dir)
    log.info("Loaded %d failed trajectories", len(trajectories))

    data_dir = Path("dataset/gaia")
    tasks_by_id = {}
    for line in (data_dir / "validation.jsonl").read_text().splitlines():
        t = json.loads(line)
        tasks_by_id[t["task_id"]] = t
    tasks_list = list(tasks_by_id.values())

    if args.only:
        keep = set(args.only.split(","))
        trajectories = [t for t in trajectories if t["instance_id"] in keep]
    else:
        trajectories = trajectories[:args.n]

    # Resume: skip instances already present in results.json
    all_results = []
    done_ids: set[str] = set()
    results_path = output_dir / "results.json"
    if results_path.exists():
        try:
            prev = json.loads(results_path.read_text())
            all_results = prev.get("results", [])
            done_ids = {r["instance_id"] for r in all_results if r.get("instance_id")}
            log.info("Resuming: %d trajectories already in results.json (skipping)",
                     len(done_ids))
        except Exception as e:
            log.warning("could not load existing results.json: %s", e)
    for traj in trajectories:
        iid = traj["instance_id"]
        if iid in done_ids:
            log.info("skip %s (already processed)", iid)
            continue
        analysis = analyses_by_iid.get(iid)
        if analysis is None:
            log.warning("no step1 analysis for %s; skip", iid)
            all_results.append({
                "instance_id": iid, "status": "no_analysis",
                "ground_truth": traj["ground_truth"],
                "wrong_answer": traj.get("final_answer"),
            })
            continue
        errors = analysis.get("error_nodes", [])

        m = re.search(r"(\d+)$", iid)
        problem_number = int(m.group(1)) if m else 0

        stem = traj["filename"].replace(".json", "")
        task = tasks_by_id.get(stem)
        if task is None and m:
            idx = int(m.group(1))
            if 0 <= idx < len(tasks_list):
                task = tasks_list[idx]
        if task is None:
            log.warning("no task for %s; skip", iid)
            continue

        per = await run_trajectory(
            traj, errors, task, output_dir, problem_number,
            args.api_url, args.model, args.timeout,
            fix_api_url=args.fix_api_url,
            fix_model=args.fix_model,
            fix_api_key=args.fix_api_key,
        )
        all_results.append(per)
        (output_dir / "results.json").write_text(
            json.dumps({"results": all_results}, indent=2, ensure_ascii=False))

    n_total = len(all_results)
    n_flipped = sum(1 for r in all_results if r.get("flip_verified"))
    (output_dir / "summary.json").write_text(json.dumps({
        "n_total": n_total, "n_flipped": n_flipped,
        "flip_rate": (n_flipped / n_total) if n_total else None,
    }, indent=2, ensure_ascii=False))
    log.info("=== SUMMARY === total=%d flipped=%d", n_total, n_flipped)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="results/owl/validation_fresh")
    p.add_argument("--output-dir", default="output/owl_counterfactual")
    p.add_argument("--phase1-file",
                   default="output/owl_counterfactual/step1_all_errors.json")
    # Agent rerun target (local vLLM by default)
    p.add_argument("--api-url", default=DEFAULT_API_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    # Fix-generation LLM (can point at OpenRouter/Claude for stronger fixes)
    p.add_argument("--fix-api-url", default=None,
                   help="If omitted, defaults to --api-url.")
    p.add_argument("--fix-model", default=None,
                   help="If omitted, defaults to --model.")
    p.add_argument("--fix-api-key", default=None,
                   help="Bearer token for fix LLM endpoint (OpenRouter, etc.).")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--only", default=None)
    args = p.parse_args()
    if args.fix_api_url is None:
        args.fix_api_url = args.api_url
    if args.fix_model is None:
        args.fix_model = args.model
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
