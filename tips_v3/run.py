"""TIPS v3 pipeline CLI.

Usage:
    python -m tips_v3.run --input-dir results/owl/validation_fresh \
                          --output-dir output/owl_counterfactual_v3 \
                          --max-trajectories 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from tips_v3 import PROMPT_VERSION, config
from tips_v3.config import SONNET_MODEL_ID, ensure_output_dirs
from tips_v3.io.ingest import iter_trajectories
from tips_v3.io.schema import Candidate, Fix, Record, SuffSetState, Trajectory
from tips_v3.io.writer import write_answer, write_difficult, write_record
from tips_v3.llm.sonnet_client import SonnetClient
from tips_v3.replay.bounded_replay import BoundedReplay
from tips_v3.stages import (
    stage2a_detect,
    stage2b_fix,
    stage3_validate,
    stage4_greedy,
    stage5_reduce,
    stage6_verify,
    stage7_filters,
    stage8_expand,
)

log = logging.getLogger("tips_v3")


def _setup_logging() -> None:
    config.ensure_output_dirs()
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(str(config.RUN_LOG)),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _build_record(
    traj: Trajectory,
    candidates: list[Candidate],
    fixes: list[Fix],
    state: SuffSetState,
    filter_result: dict,
    required_expansion: bool,
) -> Record:
    cand_by_id = {c.node_id: c for c in candidates}
    fix_by_id = {f.node_id: f for f in fixes}

    per_node = {}
    fixes_supp = {}
    for nid in state.M:
        c = cand_by_id.get(nid)
        f = fix_by_id.get(nid)
        per_node[nid] = {
            "level": c.level if c else "worker",
            "role": c.role if c else "",
            "predicted_type": c.predicted_type if c else (f.predicted_type if f else ""),
            "detection_confidence": c.confidence if c else 0.0,
            "is_uncertain_detection": bool(
                c and config.UNCERTAIN_LOWER <= c.confidence < config.UNCERTAIN_UPPER
            ),
        }
        if f is not None:
            fixes_supp[nid] = {
                "fix_text": f.new_content,
                "modified_field": f.modified_field,
                "oracle_blind": True,
            }

    return Record(
        trajectory_id=traj.trajectory_id,
        gaia_task_id=traj.gaia_task_id,
        agent_framework="OWL/Workforce",
        agent_model=traj.agent_model,
        agent_final_answer=traj.agent_final_answer,
        oracle_answer=traj.oracle_answer,
        trajectory=[n.to_dict() for n in traj.nodes],
        published_sufficient_set=list(state.M),
        per_node=per_node,
        fixes_supplementary=fixes_supp,
        validity_metadata={
            "k_final_verify": config.FINAL_VERIFY_K,
            "flip_rate_final": state.flip_rate,
            "null_fix_flip_rate": filter_result.get("null_fix_flip_rate"),
            "agent_plausibility_checked": any(f.agent_plausible for f in fixes if f.node_id in state.M),
            "leakage_audit_passed": filter_result.get("leakage_audit_passed", False),
            "required_expansion_pass": required_expansion,
            "replay_mode": "bounded",
            "prompt_version": PROMPT_VERSION,
            "sonnet_model_id": SONNET_MODEL_ID,
        },
    )


def _write_answer_file(
    traj: Trajectory,
    state: SuffSetState | None,
    outcome: str,
    fixes: list[Fix] | None = None,
    extra: dict | None = None,
) -> None:
    """Emit `answers/<tid>.json` capturing ground-truth + replay outputs.

    Written for every processed trajectory regardless of publish/difficult,
    so users can inspect what the agent produced under each fix set."""
    from tips_v3 import checkpoint as _ckpt  # local import to avoid cycles
    import sqlite3

    replay_results: list[dict] = []
    try:
        con = sqlite3.connect(str(config.REPLAY_CACHE_DB))
        for seed, final, flipped, error in con.execute(
            "SELECT seed, final_answer, flipped, error FROM cache WHERE tid=?",
            (traj.trajectory_id,),
        ):
            replay_results.append({
                "seed": seed,
                "final_answer": final,
                "flipped": bool(flipped),
                "error": error,
            })
        con.close()
    except Exception as exc:
        log.warning("answer.json: replay cache read failed for %s: %s",
                    traj.trajectory_id, exc)

    payload = {
        "trajectory_id": traj.trajectory_id,
        "ground_truth": traj.oracle_answer,
        "agent_original_final_answer": traj.agent_final_answer,
        "outcome": outcome,
        "published_sufficient_set": list(state.M) if state else [],
        "final_flip_rate": state.flip_rate if state else None,
        "replay_results": replay_results,
        "fixes": [
            {"node_id": f.node_id, "predicted_type": f.predicted_type,
             "modified_field": f.modified_field, "new_content": f.new_content}
            for f in (fixes or []) if (not state or f.node_id in state.M)
        ],
    }
    if extra:
        payload.update(extra)
    write_answer(traj.trajectory_id, payload)


def process_one(
    traj: Trajectory,
    client: SonnetClient,
    replay: BoundedReplay,
    summary: dict,
) -> None:
    log.info("start %s", traj.trajectory_id)
    required_expansion = False

    candidates = stage2a_detect.detect(traj, client)
    if not candidates:
        candidates = stage8_expand.expand_candidates(traj, [], client)
        required_expansion = True
    if not candidates:
        summary["counts"]["expansion_queue"] += 1
        write_difficult(traj.trajectory_id, {"reason": "no_candidates"})
        _write_answer_file(traj, None, "no_candidates")
        return

    fixes = stage2b_fix.propose(traj, candidates, client)
    fixes = stage3_validate.validate(traj, fixes)
    if not fixes:
        summary["counts"]["difficult"] += 1
        write_difficult(traj.trajectory_id, {"reason": "no_valid_fixes"})
        _write_answer_file(traj, None, "no_valid_fixes")
        return

    greedy_state = stage4_greedy.construct(traj, candidates, fixes, replay)
    if greedy_state.stage != "greedy" or not greedy_state.M:
        expanded = stage8_expand.expand_candidates(traj, candidates, client)
        if expanded:
            required_expansion = True
            extra_fixes = stage2b_fix.propose(traj, expanded, client)
            extra_fixes = stage3_validate.validate(traj, extra_fixes)
            candidates = candidates + expanded
            fixes = fixes + extra_fixes
            greedy_state = stage4_greedy.construct(traj, candidates, fixes, replay)
        if greedy_state.stage != "greedy" or not greedy_state.M:
            summary["counts"]["difficult"] += 1
            write_difficult(traj.trajectory_id, {"reason": "greedy_failed", "M": greedy_state.M})
            _write_answer_file(traj, greedy_state, "greedy_failed", fixes)
            return

    reduced = stage5_reduce.reduce(traj, greedy_state, fixes, replay)
    verified = stage6_verify.verify(traj, reduced, fixes, replay)
    if not stage6_verify.passed(verified):
        summary["counts"]["difficult"] += 1
        write_difficult(traj.trajectory_id, {"reason": "final_verify_failed",
                                             "flip_rate": verified.flip_rate})
        _write_answer_file(traj, verified, "final_verify_failed", fixes)
        return

    filter_result = stage7_filters.run_filters(traj, verified, fixes, client, replay, candidates=candidates)
    if not filter_result.get("leakage_audit_passed"):
        summary["counts"]["filtered_by_f2"] += 1
        write_difficult(traj.trajectory_id, {"reason": "leakage_or_null_fix", "filters": filter_result})
        _write_answer_file(traj, verified, "filtered_leakage", fixes,
                           extra={"filters": filter_result})
        return
    if not filter_result.get("seed_floor_passed"):
        summary["counts"]["difficult"] += 1
        write_difficult(traj.trajectory_id, {"reason": "seed_floor_failed"})
        _write_answer_file(traj, verified, "seed_floor_failed", fixes)
        return

    record = _build_record(traj, candidates, fixes, verified, filter_result, required_expansion)
    write_record(record)
    summary["counts"]["published"] += 1
    _write_answer_file(traj, verified, "published", fixes,
                       extra={"filters": filter_result})


def _write_summary(summary: dict) -> None:
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    config.SUMMARY_JSON.write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(config.INPUT_DIR))
    ap.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    ap.add_argument("--max-trajectories", type=int, default=0,
                    help="0 = process all")
    ap.add_argument("--only", help="comma-separated trajectory_ids to run")
    args = ap.parse_args()

    _setup_logging()
    ensure_output_dirs()

    only = set(s.strip() for s in (args.only or "").split(",") if s.strip())

    summary = {
        "run_id": str(int(time.time())),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trajectory_source": args.input_dir,
        "prompt_version": PROMPT_VERSION,
        "sonnet_model_id": SONNET_MODEL_ID,
        "counts": {
            "total_input": 0,
            "published": 0,
            "expansion_queue": 0,
            "difficult": 0,
            "filtered_by_f2": 0,
        },
    }

    client = SonnetClient()
    replay = BoundedReplay()

    n = 0
    for traj in iter_trajectories(Path(args.input_dir)):
        if only and traj.trajectory_id not in only:
            continue
        summary["counts"]["total_input"] += 1
        try:
            process_one(traj, client, replay, summary)
        except Exception as exc:
            log.exception("trajectory %s crashed: %s", traj.trajectory_id, exc)
            summary["counts"]["difficult"] += 1
            write_difficult(traj.trajectory_id, {"reason": "exception", "error": str(exc)})
            _write_answer_file(traj, None, "exception", extra={"error": str(exc)})
        n += 1
        if args.max_trajectories and n >= args.max_trajectories:
            break
        _write_summary(summary)

    _write_summary(summary)
    log.info("run complete: %s", summary["counts"])


if __name__ == "__main__":
    main()
