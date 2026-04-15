"""Build dependency graphs from Magentic-One trajectories via a single LLM call per trajectory.

This version uses per-child parent enrichment framing: for each step v, the LLM
is asked to list every earlier step that shaped it, rather than walking forward
from parents. This framing produces more complete graphs because it's easier
for the model to answer 'what shaped step v?' exhaustively than 'what are all
the descendants of step u?'.
"""

import argparse
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR = "results/magentic_one/validation"
DEFAULT_OUTPUT_DIR = "graphs"
DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ"
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TEMPERATURE = 0
CHAR_WARN_THRESHOLD = 150_000
STEP_TRUNCATE_LIMIT = 2000
STEP_TRUNCATE_HEAD = 1000
STEP_TRUNCATE_TAIL = 500
MAX_RETRIES = 3
BACKOFF_BASE = 2

# Boilerplate stripping
BOILERPLATE_MIN_LENGTH = 200
BOILERPLATE_MIN_OCCURRENCES = 3

# Degeneracy check
LINKED_LIST_FAIL_RATIO = 0.85

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are analyzing a multi-agent trajectory to identify dependency edges "
    "between steps. For every step in the trajectory, you list all earlier "
    "steps that shaped it. A step typically has multiple parents, not one. "
    "You return structured JSON only, no other text."
)

USER_PROMPT_TEMPLATE = """\
TASK: For every step v in the trajectory, identify all earlier steps that
directly shaped step v. These earlier steps are called v's PARENTS. Most
steps have MORE THAN ONE parent.

DEFINITION: Step u is a parent of step v (written u -> v) if step v's content,
action, or decision was directly shaped by information that became available
at step u. Equivalently: if step u had produced a different output, step v
would likely have produced a different output. Edges always point forward in
time (u < v).

CRITICAL: MOST STEPS HAVE MULTIPLE PARENTS.

A specialist step (WebSurfer, FileSurfer, etc.) typically has at least two
parents: the Orchestrator plan it is executing, AND the most recent tool
result or action it is building on. A step can have three, four, or more
parents when it is simultaneously:
  - executing a high-level plan from an earlier Orchestrator step,
  - building on a tool result from a different earlier step,
  - reacting to an error from yet another earlier step,
  - using specific information (URL, value, entity) that was introduced
    at yet another earlier step.

If you list only one parent for most steps, you are UNDER-COUNTING and your
output is incomplete. The graph should be dense with multiple incoming edges
per step.

COMMON FAILURE MODE TO AVOID: Do NOT produce a linked list where every step
has only one parent (step v-1). A real dependency graph has many skip edges
where a step's parent is 5, 10, or 30 positions earlier, because that is
where the relevant information, plan, or tool result originated.

RULES FOR IDENTIFYING PARENTS:

1. THE PLAN PARENT. If step v is a specialist carrying out a subtask, one of
   its parents is the most recent Orchestrator step that articulated the plan
   or subtask v is executing. This parent is often many steps earlier. If
   the plan has been re-articulated by a more recent Orchestrator step, use
   the more recent one instead.

2. THE TOOL-CALL PARENT. If step v is a tool result, its parent is the tool
   call that produced it (usually the immediately preceding step).

3. THE TOOL-RESULT PARENT. If step v is using a tool result (reading a page,
   acting on retrieved data, answering based on output), one of its parents
   is the step where that tool result first appeared.

4. THE INFORMATION PARENT. If step v references a specific URL, file path,
   value, entity name, or discovered fact, one of its parents is the most
   recent step that explicitly states or uses that specific information.
   Do NOT skip over intermediate steps to reach the original discovery step
   when a closer step already re-stated the same information.

5. THE ERROR PARENT. If step v is reacting to an error, retry, or unexpected
   output, one of its parents is the step where that error or unexpected
   output occurred. This may be many steps earlier.

6. EXCLUDE THE USER QUERY. Information that was already present in the
   original user task is not a dependency. A step referencing something from
   the query does not count as having a parent just because of that.

7. SPECIFICITY. Do not list a parent based on shared generic words or small
   numbers. The shared content must be specific: a URL, file path, computed
   value, discovered entity, articulated plan, tool result, or error message.

PROCESS (you MUST follow this order):

STEP A. Scan the trajectory once and internally note:
  - Every Orchestrator step that articulates or re-articulates a plan or
    subtask. These will be "plan parents" for many later steps.
  - Every tool call / tool result pair. Tool results have their call as a
    parent; later steps that use the result have the result as a parent.
  - Every step where specific information (URL, value, entity) first enters
    the trajectory. These will be "information parents" for later steps
    that reference that information.
  - Every error or unexpected output. These will be "error parents" for
    later steps that react to them.

STEP B. For each step v from 1 to the last step, ask yourself:
  (a) What plan is v executing? Which Orchestrator step articulated that
      plan most recently? -> add that as a parent.
  (b) Is v a tool result? -> add its tool call as a parent.
  (c) Is v using a tool result? Which step produced that result?
      -> add that as a parent.
  (d) Does v reference specific information introduced earlier?
      -> add that earlier step as a parent.
  (e) Is v reacting to an error? Which step had the error?
      -> add that as a parent.
  You should end up with MULTIPLE parents for most steps. If you end up
  with only one parent for most steps, you have missed parents — go back
  and check (a) through (e) more carefully.

STEP C. Sanity check your output:
  - Count parents per step. If most steps have only 1 parent, you have
    under-counted. Go back and look for missing plan parents (rule 1) and
    missing information parents (rule 4).
  - Count skip edges (where target - source > 1). If almost all edges are
    adjacent (target = source + 1), you have defaulted to the linked-list
    failure mode. Find the Orchestrator plans and emit their long-range
    edges.

EXAMPLES OF CORRECT AND INCORRECT EDGES:

Example 1 (CORRECT — multiple parents):
  Step 5: Orchestrator says "Now search for the article count on Nature.com"
  Step 10: WebSurfer searches Google for "Nature 2020 article count"
  Step 11: WebSurfer gets search results listing nature.com/volumes
  Step 12: WebSurfer clicks the nature.com/volumes link from step 11
  Correct edges for step 12:
    5 -> 12  (type: plan — executing the Orchestrator's search plan)
    10 -> 12 (type: tool_call — step 12 is acting on the search initiated at 10)
    11 -> 12 (type: information — the URL nature.com/volumes appeared at step 11)

Example 2 (INCORRECT — generic word match):
  Step 3: Orchestrator mentions "search"
  Step 20: WebSurfer performs a different search
  WRONG: 3 -> 20 because both mention "search" — this is a generic word, not
  a specific plan, URL, or value. Only add this edge if step 3 articulated
  the specific plan that step 20 is executing.

Example 3 (INCORRECT — linked-list default):
  Step 14: Orchestrator creates a new plan referencing results from step 8
  WRONG: 13 -> 14 as the only parent (adjacent default)
  CORRECT: 8 -> 14 (type: information — step 14 references step 8's results)
  Also correct: whichever earlier Orchestrator step set the overall strategy.

USER QUERY (for rule 6):
{query}

FULL TRAJECTORY:
{trajectory}

OUTPUT FORMAT:
Return a single JSON object with this schema:

{{
  "edges": [
    {{"source": <int>, "target": <int>, "type": "<plan|tool_call|tool_result|information|error>", "reason": "<one sentence>"}},
    ...
  ]
}}

Requirements:
- Every edge must satisfy source < target.
- Every source and target must be a valid step index in [0, {max_step}].
- Every edge must have a "type" field: one of plan, tool_call, tool_result,
  information, or error (matching rules 1-5).
- Emit edges grouped by target, in order of increasing target. Within a
  group (same target), list parents in order of increasing source.
- Keep each reason under 20 words.
- Most targets should appear in MULTIPLE edges (multiple parents). If most
  targets appear in only one edge, you have under-counted parents.
- Do not emit any duplicate edges.
- Do not artificially cap the number of parents per step. List every parent
  that is genuinely shaping the step, whether that is 1, 2, 3, 5, or more.
- Output only the JSON object, no preamble, no markdown fences, no trailing text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trajectory(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.error("Failed to load %s: %s", path, e)
        return None


def strip_boilerplate(steps: list[dict], problem_id: str) -> list[dict]:
    """Replace substrings that repeat across many steps with a short placeholder."""
    if len(steps) < BOILERPLATE_MIN_OCCURRENCES:
        return steps

    chunk_step_counts: Counter = Counter()
    chunk_size = BOILERPLATE_MIN_LENGTH
    stride = 100

    for s in steps:
        content = s.get("content", "")
        seen_in_this_step: set[str] = set()
        for i in range(0, len(content) - chunk_size + 1, stride):
            chunk = content[i:i + chunk_size]
            if chunk not in seen_in_this_step:
                seen_in_this_step.add(chunk)
                chunk_step_counts[chunk] += 1

    boilerplate_chunks = [
        c for c, count in chunk_step_counts.items()
        if count >= BOILERPLATE_MIN_OCCURRENCES
    ]

    if not boilerplate_chunks:
        return steps

    boilerplate_chunks.sort(key=len, reverse=True)

    log.info(
        "%s: stripping %d boilerplate chunks (largest %d chars)",
        problem_id, len(boilerplate_chunks), len(boilerplate_chunks[0]),
    )

    cleaned_steps = []
    total_chars_removed = 0
    for s in steps:
        content = s.get("content", "")
        original_len = len(content)
        for chunk in boilerplate_chunks:
            if chunk in content:
                content = content.replace(chunk, "[navigation chrome]")
        total_chars_removed += original_len - len(content)
        new_s = dict(s)
        new_s["content"] = content
        cleaned_steps.append(new_s)

    log.info("%s: removed %d total boilerplate chars", problem_id, total_chars_removed)
    return cleaned_steps


def format_trajectory(steps: list[dict], problem_id: str) -> str:
    lines = []
    for s in steps:
        idx = s["step"]
        agent = s["agent"]
        content = s["content"]
        if len(content) > STEP_TRUNCATE_LIMIT:
            log.info("Truncating step %d of %s (%d chars)", idx, problem_id, len(content))
            content = (
                content[:STEP_TRUNCATE_HEAD]
                + "\n...[truncated]...\n"
                + content[-STEP_TRUNCATE_TAIL:]
            )
        lines.append(f"[Step {idx}] {agent}: {content}")
    return "\n".join(lines)


def build_prompt(query: str, trajectory_text: str, max_step: int) -> str:
    return USER_PROMPT_TEMPLATE.format(
        query=query,
        trajectory=trajectory_text,
        max_step=max_step,
    )


def call_llm(
    api_url: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Call the vLLM OpenAI-compatible endpoint. Returns the full response JSON."""
    with httpx.Client(timeout=600) as client:
        resp = client.post(
            api_url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return resp.json()


def extract_json(text: str) -> dict | None:
    """Try to parse text as JSON; fall back to extracting last balanced {...} block."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Prefer the last balanced block containing "edges"
    matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            obj = json.loads(m.group())
            if "edges" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def validate_edges(edges: list, num_steps: int) -> list[dict]:
    """Validate and deduplicate edges. Returns cleaned list."""
    valid = []
    seen = set()
    dropped = 0
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        reason = e.get("reason", "")
        if not isinstance(src, int) or not isinstance(tgt, int):
            dropped += 1
            continue
        if src < 0 or src >= num_steps or tgt < 0 or tgt >= num_steps:
            dropped += 1
            continue
        if src >= tgt:
            dropped += 1
            continue
        key = (src, tgt)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        edge = {"source": src, "target": tgt, "reason": reason}
        edge_type = e.get("type", "")
        if edge_type in ("plan", "tool_call", "tool_result", "information", "error"):
            edge["type"] = edge_type
        valid.append(edge)
    if dropped:
        log.warning("Dropped %d invalid/duplicate edges", dropped)
    return valid


def compute_edge_stats(edges: list[dict], num_steps: int) -> dict:
    """Compute graph quality statistics.

    Under option-2 framing we care most about in-degree distribution (parents
    per child). A healthy graph has most non-root steps with 2+ parents.
    """
    if not edges:
        return {
            "num_edges": 0,
            "num_adjacent_edges": 0,
            "num_skip_edges": 0,
            "adjacent_ratio": 0.0,
            "density": 0.0,
            "max_skip_distance": 0,
            "avg_skip_distance": 0.0,
            "avg_parents_per_child": 0.0,
            "multi_parent_fraction": 0.0,
            "max_in_degree": 0,
            "max_out_degree": 0,
            "is_linked_list": False,
            "is_under_counted": False,
        }

    adjacent = sum(1 for e in edges if e["target"] - e["source"] == 1)
    skip = len(edges) - adjacent
    skip_distances = [e["target"] - e["source"] for e in edges if e["target"] - e["source"] > 1]
    adjacent_ratio = adjacent / len(edges) if edges else 0.0
    max_possible = num_steps * (num_steps - 1) / 2 if num_steps > 1 else 1
    density = len(edges) / max_possible if max_possible > 0 else 0.0

    # Parents-per-child and fan-out analysis
    in_degree: dict[int, int] = defaultdict(int)
    out_degree: dict[int, int] = defaultdict(int)
    for e in edges:
        in_degree[e["target"]] += 1
        out_degree[e["source"]] += 1

    targets_with_edges = list(in_degree.keys())
    num_targets = len(targets_with_edges)
    avg_parents = sum(in_degree.values()) / num_targets if num_targets else 0.0
    multi_parent = sum(1 for t in targets_with_edges if in_degree[t] >= 2)
    multi_parent_fraction = multi_parent / num_targets if num_targets else 0.0

    # Under-counted heuristic: healthy graphs have at least ~50% of targets
    # with 2+ parents. If most targets have only 1 parent, the model likely
    # missed plan parents and information parents.
    is_under_counted = num_targets > 5 and multi_parent_fraction < 0.3

    return {
        "num_edges": len(edges),
        "num_adjacent_edges": adjacent,
        "num_skip_edges": skip,
        "adjacent_ratio": round(adjacent_ratio, 4),
        "density": round(density, 4),
        "max_skip_distance": max(skip_distances) if skip_distances else 0,
        "avg_skip_distance": round(sum(skip_distances) / len(skip_distances), 2) if skip_distances else 0.0,
        "avg_parents_per_child": round(avg_parents, 2),
        "multi_parent_fraction": round(multi_parent_fraction, 4),
        "max_in_degree": max(in_degree.values()) if in_degree else 0,
        "max_out_degree": max(out_degree.values()) if out_degree else 0,
        "is_linked_list": adjacent_ratio >= LINKED_LIST_FAIL_RATIO,
        "is_under_counted": is_under_counted,
    }


# ---------------------------------------------------------------------------
# Main per-trajectory processing
# ---------------------------------------------------------------------------

def process_trajectory(
    path: Path,
    problem_id: str,
    api_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    strip_chrome: bool,
) -> dict:
    """Process one trajectory file. Returns the result dict for output."""
    data = load_trajectory(path)
    if data is None:
        return {"problem_id": problem_id, "status": "failed", "error": "load_failed"}

    query = data.get("query", "")
    steps = [s for s in data.get("failure_log", []) if s["agent"] != "system"]
    num_steps = len(steps)

    if num_steps == 0:
        log.warning("%s has no steps", problem_id)
        return {"problem_id": problem_id, "status": "failed", "error": "no_steps", "num_steps": 0}

    for i, s in enumerate(steps):
        s["step"] = i
    max_step = num_steps - 1

    if strip_chrome:
        steps = strip_boilerplate(steps, problem_id)

    trajectory_text = format_trajectory(steps, problem_id)
    user_prompt = build_prompt(query, trajectory_text, max_step)

    if len(user_prompt) > CHAR_WARN_THRESHOLD:
        log.warning(
            "%s prompt is %d chars (>%d), may exceed token budget",
            problem_id, len(user_prompt), CHAR_WARN_THRESHOLD,
        )

    raw_response = None
    parsed = None
    input_tokens = 0
    output_tokens = 0
    latency = 0.0
    attempts = 0

    for attempt in range(MAX_RETRIES):
        attempts = attempt + 1
        try:
            t0 = time.time()
            resp = call_llm(api_url, model, SYSTEM_PROMPT, user_prompt, max_tokens, temperature)
            latency = time.time() - t0

            usage = resp.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            choice = resp["choices"][0]
            raw_response = choice["message"]["content"]

            if choice.get("finish_reason") == "length":
                log.error("%s: response truncated (hit max_tokens)", problem_id)
                return {
                    "problem_id": problem_id,
                    "num_steps": num_steps,
                    "model_used": f"vllm on {model}",
                    "edges": [],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_seconds": round(latency, 2),
                    "raw_response": raw_response,
                    "status": "failed",
                    "error": "response_truncated",
                }

            parsed = extract_json(raw_response)
            if parsed is not None:
                break
            log.warning("%s: JSON parse failed (attempt %d), retrying", problem_id, attempts)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 429 or code >= 500:
                wait = BACKOFF_BASE ** attempt
                log.warning("%s: HTTP %d, retrying in %ds", problem_id, code, wait)
                time.sleep(wait)
            else:
                log.error("%s: HTTP %d (non-retryable)", problem_id, code)
                break
        except Exception as e:
            log.error("%s: LLM call failed: %s", problem_id, e)
            wait = BACKOFF_BASE ** attempt
            time.sleep(wait)

    if parsed is None:
        log.error("%s: failed to get valid JSON after %d attempts", problem_id, attempts)
        return {
            "problem_id": problem_id,
            "num_steps": num_steps,
            "model_used": f"vllm on {model}",
            "edges": [],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": round(latency, 2),
            "raw_response": raw_response or "",
            "status": "failed",
            "error": "json_parse_failed",
        }

    raw_edges = parsed.get("edges", [])
    edges = validate_edges(raw_edges, num_steps)
    stats = compute_edge_stats(edges, num_steps)

    warnings = []
    if stats["is_linked_list"]:
        warnings.append(
            f"linked_list_degeneracy: {stats['adjacent_ratio']:.0%} of edges are adjacent"
        )
        log.warning(
            "%s: LINKED LIST DETECTED (%.0f%% adjacent edges)",
            problem_id, stats['adjacent_ratio'] * 100,
        )
    if stats["is_under_counted"]:
        warnings.append(
            f"under_counted_parents: only {stats['multi_parent_fraction']:.0%} of targets have 2+ parents"
        )
        log.warning(
            "%s: PARENT UNDER-COUNT (only %.0f%% of targets have 2+ parents, avg %.1f)",
            problem_id, stats['multi_parent_fraction'] * 100, stats['avg_parents_per_child'],
        )

    return {
        "problem_id": problem_id,
        "num_steps": num_steps,
        "model_used": f"vllm on {model}",
        "edges": edges,
        "stats": stats,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": round(latency, 2),
        "raw_response": raw_response,
        "status": "success",
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build dependency graphs from Magentic-One trajectories")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--no-strip-chrome", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in input_dir.glob("*.json") if f.stem != "summary"])
    log.info("Found %d trajectory files in %s", len(files), input_dir)

    results = []
    wall_start = time.time()

    for idx, fpath in enumerate(files):
        problem_id = f"problem_{idx + 1:02d}"
        log.info("Processing %s (%s)", problem_id, fpath.name)

        result = process_trajectory(
            fpath, problem_id,
            args.api_url, args.model,
            args.max_tokens, args.temperature,
            strip_chrome=not args.no_strip_chrome,
        )

        out_path = output_dir / f"{problem_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        stats = result.get("stats", {})
        log.info(
            "  Written: %s  edges=%d  skip=%d  avg_parents=%.1f  multi_parent=%.0f%%  status=%s",
            out_path,
            stats.get("num_edges", 0),
            stats.get("num_skip_edges", 0),
            stats.get("avg_parents_per_child", 0.0),
            stats.get("multi_parent_fraction", 0.0) * 100,
            result.get("status"),
        )

        results.append(result)

    total_wall = time.time() - wall_start

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    linked_list_degenerate = [r for r in results if r.get("stats", {}).get("is_linked_list", False)]
    under_counted = [r for r in results if r.get("stats", {}).get("is_under_counted", False)]

    per_problem = []
    for r in results:
        stats = r.get("stats", {})
        per_problem.append({
            "problem_id": r["problem_id"],
            "num_steps": r.get("num_steps", 0),
            "num_edges": stats.get("num_edges", 0),
            "num_skip_edges": stats.get("num_skip_edges", 0),
            "adjacent_ratio": stats.get("adjacent_ratio", 0.0),
            "density": stats.get("density", 0.0),
            "avg_parents_per_child": stats.get("avg_parents_per_child", 0.0),
            "multi_parent_fraction": stats.get("multi_parent_fraction", 0.0),
            "max_in_degree": stats.get("max_in_degree", 0),
            "max_out_degree": stats.get("max_out_degree", 0),
            "is_linked_list": stats.get("is_linked_list", False),
            "is_under_counted": stats.get("is_under_counted", False),
            "status": r.get("status", "unknown"),
        })

    summary = {
        "num_problems": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "linked_list_degenerate": len(linked_list_degenerate),
        "under_counted": len(under_counted),
        "total_steps_analyzed": sum(r.get("num_steps", 0) for r in results),
        "total_edges_extracted": sum(len(r.get("edges", [])) for r in results),
        "total_skip_edges": sum(r.get("stats", {}).get("num_skip_edges", 0) for r in results),
        "total_input_tokens": sum(r.get("input_tokens", 0) for r in results),
        "total_output_tokens": sum(r.get("output_tokens", 0) for r in results),
        "total_wall_time_seconds": round(total_wall, 2),
        "per_problem": per_problem,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary written to %s", summary_path)
    log.info(
        "Done: %d/%d successful, %d edges (%d skip), %d linked-list, %d under-counted, %.1fs",
        len(successful), len(results),
        summary["total_edges_extracted"],
        summary["total_skip_edges"],
        summary["linked_list_degenerate"],
        summary["under_counted"],
        total_wall,
    )

    if linked_list_degenerate:
        log.warning(
            "LINKED LIST DEGENERACY in %d/%d: %s",
            len(linked_list_degenerate), len(results),
            [r["problem_id"] for r in linked_list_degenerate],
        )
    if under_counted:
        log.warning(
            "PARENT UNDER-COUNT in %d/%d: %s",
            len(under_counted), len(results),
            [r["problem_id"] for r in under_counted],
        )


if __name__ == "__main__":
    main()
