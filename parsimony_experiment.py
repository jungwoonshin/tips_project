"""Parsimony frontier experiment: percentile k-sweep over error steps ranked by
trajectory order (earliest-first), excluding the terminal answer-emitting step.

Output: per-problem results (k*, fixes applied, flip status) + a labeled dataset
of multi-error-step ground truth for the trajectories that flipped.

Protocol per trajectory (n = number of candidate error steps after filtering):
  1. Filter: drop any error node at the trajectory's terminal step
     (that step emits the final answer; fixing it trivially flips).
  2. Rank remaining nodes by step index ascending (earliest-first).
  3. Sweep k at {1, ceil(n*0.33), ceil(n*0.66), n} (unique, ascending):
       - Apply top-k fixes (cumulatively), rerun from after the last fixed step.
       - If flipped -> record k*, stop.
     If the sweep exhausts without a flip -> unsalvageable.
  4. Confirmation rerun at k* (1 additional trial).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path

from linear_identify_and_fix import (
    DEFAULT_API_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT, MAX_TOKENS,
    FIX_SYSTEM, FIX_USER,
    call_llm_raw, format_step_content, format_trajectory,
    load_failed_trajectories,
    _build_history_messages, _rerun_single,
)
from trajectory.evaluator import evaluate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parsimony")


# ---------------------------------------------------------------------------
# Error-node ordering: earliest-first, exclude terminal answer-emitting step
# ---------------------------------------------------------------------------

def rank_error_steps(traj: dict, error_nodes: list):
    """Rank error nodes by step index ascending; drop the terminal step.

    The terminal step (max step index in the trajectory) is where the final
    answer is emitted. Fixing it rewrites the answer directly, which trivially
    flips without isolating any causal decision upstream, so it is excluded
    from the candidate pool.
    """
    if not traj["steps"]:
        return []
    terminal_step = max(s["step"] for s in traj["steps"])
    ranked = []
    for n in error_nodes:
        if n["step"] >= terminal_step:
            continue
        step = next((s for s in traj["steps"] if s["step"] == n["step"]), None)
        if step is None:
            continue
        ranked.append({"node": n, "step": n["step"]})
    ranked.sort(key=lambda r: r["step"])
    return ranked


# ---------------------------------------------------------------------------
# Fix generation for a set of error steps
# ---------------------------------------------------------------------------

def generate_fixes(traj: dict, nodes_by_step_asc: list, api_url: str, model: str):
    """Generate fixes for the given error nodes in trajectory-order.

    Returns a list of fix dicts (one per node) each containing:
      step, agent, error_type, original_content, corrected_content.
    Earlier fixes in the list are applied as prior_context for later fixes.
    """
    steps = traj["steps"]
    applied = {}
    out = []
    for node in nodes_by_step_asc:
        step_idx = node["step"]
        agent = node["agent"]

        prior = []
        for s in steps:
            if s["step"] >= step_idx:
                break
            ns = dict(s)
            if ns["step"] in applied:
                ns["content"] = applied[ns["step"]]
            prior.append(ns)

        prior_ctx = format_trajectory(prior)
        step_data = next(s for s in steps if s["step"] == step_idx)
        original = format_step_content(step_data["content"])

        sys_prompt = FIX_SYSTEM.format(
            agent_name=agent,
            agent_system_prompt=traj["agent_prompts"].get(agent, ""),
        )
        usr_prompt = FIX_USER.format(
            prior_context=prior_ctx,
            original_content=original,
            error_type=node.get("error_type", "UNKNOWN"),
            wrong_final_answer=traj.get("final_answer") or "(no answer produced)",
            agent_name=agent,
            original_length=len(step_data["content"]),
        )
        fix_tokens = min(MAX_TOKENS, max(512, len(step_data["content"]) * 2))
        raw, _ = call_llm_raw(api_url, model, sys_prompt, usr_prompt, max_tokens=fix_tokens)
        corrected = raw.strip()
        applied[step_idx] = corrected
        out.append({
            "step": step_idx,
            "agent": agent,
            "error_type": node.get("error_type"),
            "original_content": step_data["content"],
            "corrected_content": corrected,
        })
    return out


# ---------------------------------------------------------------------------
# Rerun wrapper
# ---------------------------------------------------------------------------

async def _single_rerun(traj, fixes, task, runner, rerun_dir, problem_number, label,
                         timeout, checkpoint_dir=None):
    from trajectory.writer import TrajectoryWriter
    from runners.magentic_one_runner import AGENT_DEFINITIONS

    last_fixed = max(f["step"] for f in fixes)
    fix_records = [{
        "step": f["step"],
        "agent": f.get("agent"),
        "original_content": f.get("original_content", ""),
        "corrected_content": f["corrected_content"],
        "status": "success",
    } for f in fixes]
    history = _build_history_messages(traj["steps"], fix_records, last_fixed)

    writer = TrajectoryWriter(
        task=task, framework="magentic-one-parsimony", model=runner.model,
        output_dir=str(rerun_dir), instance_id=traj["instance_id"] + f"_{label}",
        problem_number=problem_number, agents=AGENT_DEFINITIONS,
    )
    writer.enable_live_log()

    t0 = time.time()
    try:
        rerun_task = asyncio.ensure_future(
            _rerun_single(
                task, history, runner, writer, problem_id=label,
                checkpoint_dir=checkpoint_dir, fixes=fix_records, last_fixed_step=last_fixed,
            ),
        )
        loop = asyncio.get_event_loop()
        watchdog = threading.Timer(timeout, lambda: loop.call_soon_threadsafe(rerun_task.cancel))
        watchdog.start()
        try:
            final_answer = await rerun_task
        finally:
            watchdog.cancel()
        latency = time.time() - t0
        writer.finalize(final_answer=final_answer)
        return {"final_answer": final_answer, "latency_seconds": round(latency, 2),
                "status": "success", "last_fixed_step": last_fixed}
    except (asyncio.TimeoutError, asyncio.CancelledError):
        writer.finalize(error="Timeout")
        return {"final_answer": None, "status": "timeout", "last_fixed_step": last_fixed}
    except Exception as e:
        writer.finalize(error=str(e))
        return {"final_answer": None, "status": "failed", "error": str(e),
                "last_fixed_step": last_fixed}


# ---------------------------------------------------------------------------
# k-levels: percentile sample of [1..n] -> {1, ceil(n*0.33), ceil(n*0.66), n}
# ---------------------------------------------------------------------------

def compute_k_levels(n: int) -> list:
    """Percentile sample of [1..n]. Returns unique sorted ascending."""
    import math
    if n <= 0:
        return []
    return sorted({1, math.ceil(n * 0.33), math.ceil(n * 0.66), n})


# ---------------------------------------------------------------------------
# Percentile k-sweep search for one trajectory
# ---------------------------------------------------------------------------

async def run_trajectory(traj, error_nodes, task, runner, output_dir,
                          problem_number, api_url, model, timeout,
                          checkpoint_root=None):
    instance_id = traj["instance_id"]
    log.info("=== %s (%s) ===", instance_id, f"problem_{problem_number:02d}")
    log.info("  num_steps=%d  n_errors=%d", len(traj["steps"]), len(error_nodes))
    checkpoint_dir = None
    if checkpoint_root is not None:
        cp = Path(checkpoint_root) / f"problem_{problem_number:02d}"
        if cp.exists():
            checkpoint_dir = cp

    wrong_answer = traj.get("final_answer") or ""
    ground_truth = traj["ground_truth"]
    ranked = rank_error_steps(traj, error_nodes)
    ordered_nodes = [r["node"] for r in ranked]
    n = len(ordered_nodes)
    levels = compute_k_levels(n)
    log.info("  ordering (earliest-first, excl. terminal): %s",
             [r["step"] for r in ranked])
    log.info("  k-levels (1/33/66/100 pct): %s", levels)

    result = {
        "instance_id": instance_id,
        "problem_number": problem_number,
        "num_steps": len(traj["steps"]),
        "ground_truth": ground_truth,
        "wrong_answer": wrong_answer,
        "ordering": [{"step": r["step"]} for r in ranked],
        "k_levels_tested": levels,
        "levels": [],
        "k_star": None,
        "flip_verified": False,
        "confirmation_flipped": None,
        "minimal_critical_set": [],
        "fixes_at_kstar": [],
    }

    async def _test_level(k: int, tag: str):
        """Run fix + rerun at level k. Record level. Return (flipped, fixes)."""
        selected = ordered_nodes[:k]
        selected_sorted = sorted(selected, key=lambda x: x["step"])
        log.info("  -> level k=%d [%s], steps=%s",
                 k, tag, [x["step"] for x in selected_sorted])
        fixes = generate_fixes(traj, selected_sorted, api_url, model)
        rerun_dir = output_dir / f"reruns_k{k}"
        rerun_dir.mkdir(parents=True, exist_ok=True)
        rerun = await _single_rerun(traj, fixes, task, runner, rerun_dir,
                                     problem_number, label=f"k{k}", timeout=timeout,
                                     checkpoint_dir=checkpoint_dir)
        flipped = (rerun.get("status") == "success"
                   and evaluate_answer(rerun.get("final_answer"), ground_truth))
        level_rec = {
            "k": k,
            "tag": tag,
            "selected_steps": [x["step"] for x in selected_sorted],
            "rerun_answer": rerun.get("final_answer"),
            "rerun_status": rerun.get("status"),
            "flipped": bool(flipped),
            "last_fixed_step": rerun.get("last_fixed_step"),
            "latency_seconds": rerun.get("latency_seconds"),
            "fixes": fixes,
        }
        result["levels"].append(level_rec)
        log.info("    rerun_answer=%s  flipped=%s", rerun.get("final_answer"), flipped)
        return bool(flipped), fixes

    if n == 0:
        log.info("  => SKIPPED (0 candidate error nodes after filtering)")
        return result

    # Percentile sweep (ascending). First flip wins; exhaustion = unsalvageable.
    best_k, best_fixes = None, None
    for k in levels:
        flipped_k, fixes_k = await _test_level(k, tag="percentile-sweep")
        if flipped_k:
            best_k, best_fixes = k, fixes_k
            break

    if best_k is None:
        log.info("  => UNSALVAGEABLE (no flip across k-levels %s)", levels)
        return result

    result["k_star"] = best_k
    result["flip_verified"] = True
    result["minimal_critical_set"] = sorted(
        x["step"] for x in ordered_nodes[:best_k]
    )
    result["fixes_at_kstar"] = best_fixes

    # Confirmation rerun at k*
    log.info("  -> confirmation rerun at k*=%d", best_k)
    conf_dir = output_dir / "reruns_confirm"
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf = await _single_rerun(traj, best_fixes, task, runner, conf_dir,
                                problem_number, label=f"confirm_k{best_k}",
                                timeout=timeout, checkpoint_dir=checkpoint_dir)
    conf_flipped = (conf.get("status") == "success"
                    and evaluate_answer(conf.get("final_answer"), ground_truth))
    result["confirmation_flipped"] = bool(conf_flipped)
    result["confirmation_answer"] = conf.get("final_answer")
    log.info("    confirmation_answer=%s  flipped=%s",
             conf.get("final_answer"), conf_flipped)

    log.info("  => FLIPPED at k*=%d (steps %s), confirmation=%s",
             result["k_star"], result["minimal_critical_set"],
             result["confirmation_flipped"])
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

async def main_async(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "parsimony.log"
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    log.info("Logging intermediate progress to %s", log_path)

    step1_path = Path(args.phase1_file)
    step1 = json.loads(step1_path.read_text())
    analyses_by_iid = {a["instance_id"]: a for a in step1.get("analyses", [])}

    # Count total/pass/fail over the whole input directory (for pass_rate)
    total_trajectories = 0
    originally_correct = 0
    for fp in sorted(input_dir.glob("*.json")):
        if fp.stem == "summary":
            continue
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        total_trajectories += 1
        if d.get("task_outcome") == "success":
            originally_correct += 1

    trajectories = load_failed_trajectories(input_dir)
    log.info("Loaded %d failed trajectories (of %d total, %d originally correct)",
             len(trajectories), total_trajectories, originally_correct)

    # Load GAIA tasks
    data_dir = Path("dataset/gaia")
    tasks_by_id = {}
    for line in (data_dir / "validation.jsonl").read_text().splitlines():
        t = json.loads(line)
        tasks_by_id[t["task_id"]] = t

    # Filter: process only the requested instance(s) if specified
    if args.only:
        keep = set(args.only.split(","))
        trajectories = [t for t in trajectories if t["instance_id"] in keep]
        log.info("Filtered to %d trajectories: %s", len(trajectories),
                 [t["instance_id"] for t in trajectories])

    from runners.magentic_one_runner import MagenticOneRunner
    runner = MagenticOneRunner(
        data_dir=str(data_dir), output_dir=str(output_dir / "unused"),
        model=args.model,
    )

    all_results = []
    for traj in trajectories:
        iid = traj["instance_id"]
        analysis = analyses_by_iid.get(iid)
        if analysis is None:
            log.warning("  no step1 analysis for %s, skipping", iid)
            continue
        errors = analysis.get("error_nodes", [])
        if not errors:
            log.info("  %s has 0 error nodes, skipping", iid)
            all_results.append({
                "instance_id": iid, "num_steps": analysis.get("num_steps"),
                "ground_truth": traj["ground_truth"],
                "wrong_answer": traj.get("final_answer"),
                "ordering": [], "levels": [], "k_star": None,
                "flip_verified": False, "skip_reason": "no_error_nodes",
            })
            continue

        # Derive problem number from instance id suffix
        m = re.search(r"(\d+)$", iid)
        problem_number = int(m.group(1)) if m else 0
        task = tasks_by_id.get(traj["filename"].replace(".json", ""))
        if task is None:
            task_candidates = [t for k, t in tasks_by_id.items()
                               if k == traj["filename"].replace(".json", "")]
            if task_candidates:
                task = task_candidates[0]
        if task is None:
            import re as _re
            _m = _re.search(r"(\d+)$", iid)
            if _m:
                idx = int(_m.group(1))
                tasks_list = list(tasks_by_id.values())
                if 0 <= idx < len(tasks_list):
                    task = tasks_list[idx]

        if task is None:
            log.warning("  no task for %s, skipping", iid)
            continue

        per = await run_trajectory(
            traj, errors, task, runner, output_dir,
            problem_number, args.api_url, args.model, args.timeout,
            checkpoint_root=input_dir / "checkpoints",
        )
        all_results.append(per)

        # Save incrementally after each trajectory
        (output_dir / "results.json").write_text(
            json.dumps({"results": all_results}, indent=2, ensure_ascii=False))

    await runner.cleanup()

    # Build the final labeled dataset (only flipped trajectories)
    dataset = []
    for r in all_results:
        if not r.get("flip_verified"):
            continue
        dataset.append({
            "instance_id": r["instance_id"],
            "problem_number": r["problem_number"],
            "query_ground_truth": r["ground_truth"],
            "wrong_final_answer": r["wrong_answer"],
            "num_steps": r["num_steps"],
            "minimal_critical_set": r["minimal_critical_set"],
            "k_star": r["k_star"],
            "confirmation_flipped": r["confirmation_flipped"],
            "ordering_policy": "step_index_ascending_excl_terminal",
            "ordered_candidates": [o["step"] for o in r["ordering"]],
            "labeled_error_steps": [
                {
                    "step": f["step"],
                    "agent": f["agent"],
                    "error_type": f["error_type"],
                    "corrected_content": f["corrected_content"],
                }
                for f in r["fixes_at_kstar"]
            ],
        })
    (output_dir / "labeled_dataset.json").write_text(
        json.dumps({"dataset": dataset,
                    "n_included": len(dataset),
                    "n_total_analyzed": len(all_results)},
                   indent=2, ensure_ascii=False))
    log.info("Saved labeled dataset: %d / %d trajectories included",
             len(dataset), len(all_results))

    # ---- Primary benchmark metrics --------------------------------------
    # pass_rate        = originally correct / total
    # salvageability_rate = flipped / all failures (strict denominator)
    # median_k_star    = median k* across flipped trajectories
    n_failures = max(0, total_trajectories - originally_correct)
    flipped_results = [r for r in all_results if r.get("flip_verified")]
    k_stars = sorted(r["k_star"] for r in flipped_results if r.get("k_star") is not None)

    def _median(xs):
        if not xs:
            return None
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2

    pass_rate = (originally_correct / total_trajectories) if total_trajectories else None
    salvageability_rate = (len(flipped_results) / n_failures) if n_failures else None
    median_k_star = _median(k_stars)

    summary = {
        "n_total_trajectories": total_trajectories,
        "n_originally_correct": originally_correct,
        "n_failures": n_failures,
        "n_failures_with_error_nodes": sum(
            1 for r in all_results if r.get("skip_reason") != "no_error_nodes"),
        "n_failures_no_error_nodes": sum(
            1 for r in all_results if r.get("skip_reason") == "no_error_nodes"),
        "n_flipped": len(flipped_results),
        "n_unsalvageable": len(all_results) - len(flipped_results),
        "pass_rate": pass_rate,
        "salvageability_rate": salvageability_rate,
        "median_k_star": median_k_star,
        "k_star_distribution": k_stars,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("=== SUMMARY ===")
    log.info("  pass_rate            = %s  (%d / %d)",
             f"{pass_rate:.3f}" if pass_rate is not None else "N/A",
             originally_correct, total_trajectories)
    log.info("  salvageability_rate  = %s  (%d / %d failures)",
             f"{salvageability_rate:.3f}" if salvageability_rate is not None else "N/A",
             len(flipped_results), n_failures)
    log.info("  median_k_star        = %s  (over %d flipped)",
             median_k_star if median_k_star is not None else "N/A",
             len(flipped_results))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="results/magentic_one/validation")
    p.add_argument("--output-dir", default="output/parsimony")
    p.add_argument("--phase1-file", default="output/ablation_clustered/step1_all_errors.json")
    p.add_argument("--api-url", default=DEFAULT_API_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--only", default=None,
                   help="Comma-separated instance_ids to process")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
